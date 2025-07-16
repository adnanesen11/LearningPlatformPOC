import os
import json
import datetime
import subprocess
import assemblyai as aai
from pytubefix import YouTube
from fpdf import FPDF
import os
from dotenv import load_dotenv
import boto3
from langchain_community.vectorstores import FAISS
from FlagEmbedding import BGEM3FlagModel
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings
from langchain_aws.chat_models import ChatBedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional
import re
import numpy as np
from clip_manager import VideoClipManager

# === SETUP ===
aai.settings.api_key = "8a010ac1fdb74105bca2aa651b32a3e2"  # Replace for production

load_dotenv()

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
)

class BGE_M3_Embedder:
    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        print("🔍 Embedding with BGE-M3...")
        return self.model.encode(texts, max_length=8192)["dense_vecs"]

bge_m3_embedder = BGE_M3_Embedder()

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        print("ffmpeg not found.")
        return False

def time_str_to_seconds(time_str: str) -> float:
    """Convert time string (H:MM:SS.microseconds) to seconds"""
    try:
        # Handle format like "0:00:03.600000"
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except:
        return 0.0

def seconds_to_time_str(seconds: float) -> str:
    """Convert seconds to FFmpeg time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def extract_video_clip(video_file: str, start_time: float, end_time: float, output_path: str) -> bool:
    """Extract video clip using FFmpeg"""
    try:
        start_str = seconds_to_time_str(start_time)
        duration = end_time - start_time
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', start_str,
            '-i', video_file,
            '-t', str(duration),
            '-c', 'copy',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error extracting video clip: {e}")
        return False

class TimestampedDocument(Document):
    """Extended Document class that includes timestamp metadata"""
    def __init__(self, page_content: str, metadata: dict = None, video_file: str = None, 
                 start_time: str = None, end_time: str = None):
        # Prepare metadata with timestamp info
        enhanced_metadata = metadata or {}
        if video_file:
            enhanced_metadata.update({
                'video_file': video_file,
                'start_time': start_time,
                'end_time': end_time
            })
        
        super().__init__(page_content=page_content, metadata=enhanced_metadata)

class BGEWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts)['dense_vecs']

    def embed_query(self, text):
        return self.model.encode([text])['dense_vecs'][0]

class EnhancedQASystem:
    def __init__(self, save_clips: bool = True, similarity_threshold: float = 0.65, use_bedrock_quality: bool = True, use_mmr: bool = False):
        self.db = None
        self.qa_chain = None
        self.video_files = {}  # Map video names to file paths
        self.sentence_data = {}  # Map video names to sentence data
        self.save_clips = save_clips
        self.clip_manager = VideoClipManager() if save_clips else None
        self.similarity_threshold = similarity_threshold  # Minimum similarity for clip generation
        self.bge_model = None  # For answer-aware clip selection
        self.use_bedrock_quality = use_bedrock_quality  # Use Bedrock KB quality retrieval
        self.use_mmr = use_mmr  # Use Maximal Marginal Relevance for diversity
        self.bedrock_similarity_threshold = 0.15  # Lower threshold to match Bedrock KB behavior
        self.claude_model = None  # Direct Claude model for custom prompts
        
    def load_timestamped_content(self, sentences_file: str, video_file: str):
        """Load timestamped sentences and create enhanced documents"""
        print(f"📚 Loading timestamped content from {sentences_file}")
        
        # Load sentence data
        with open(sentences_file, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        
        # Store for later use
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        self.video_files[video_name] = video_file
        self.sentence_data[video_name] = sentences
        
        # Create documents with timestamp metadata
        documents = []
        
        # Group sentences into chunks (e.g., every 3-5 sentences)
        chunk_size = 4  # Number of sentences per chunk
        for i in range(0, len(sentences), chunk_size):
            chunk_sentences = sentences[i:i + chunk_size]
            
            # Combine text from sentences in chunk
            chunk_text = " ".join([s['text'] for s in chunk_sentences])
            
            # Get start time from first sentence, end time from last sentence
            start_time = chunk_sentences[0]['start']
            end_time = chunk_sentences[-1]['end']
            
            # Create timestamped document
            doc = TimestampedDocument(
                page_content=chunk_text,
                video_file=video_file,
                start_time=start_time,
                end_time=end_time,
                metadata={
                    'chunk_index': i // chunk_size,
                    'sentence_count': len(chunk_sentences),
                    'video_name': video_name
                }
            )
            documents.append(doc)
        
        return documents
    
    def create_vector_store(self, documents: List[TimestampedDocument]):
        """Create FAISS vector store from timestamped documents"""
        print("🔍 Creating vector store with timestamp metadata...")
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self.bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        embedding = BGEWrapper(self.bge_model)
        
        self.db = FAISS.from_texts(texts=texts, embedding=embedding, metadatas=metadatas)
        return self.db
    
    def setup_qa_chain(self):
        """Setup the QA chain"""
        retriever = self.db.as_retriever(search_kwargs={"k": 5})  # Get top 5 relevant chunks
        self.claude_model = ChatBedrock(
            client=bedrock_client,
            model_id="arn:aws:bedrock:us-east-1:225989333617:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            provider="anthropic"
        )
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.claude_model, retriever=retriever, return_source_documents=True)
    
    def get_filtered_chunks(self, query: str, k: int = 25, similarity_threshold: float = 0.68) -> List[Tuple[Document, float]]:
        """Get top chunks with cosine similarity filtering - Bedrock KB quality"""
        if not self.db:
            return []
        
        print(f"🔍 Bedrock Quality Retrieval: Getting {k} candidates, filtering by similarity > {similarity_threshold}")
        
        # Get k=25 candidates with similarity scores
        candidate_docs = self.db.similarity_search_with_score(query, k=k)
        
        print(f"📊 Retrieved {len(candidate_docs)} candidate chunks")
        
        # Filter by cosine similarity > threshold
        filtered_docs = []
        for doc, score in candidate_docs:
            # FAISS with cosine similarity returns L2 distance of normalized vectors
            # Convert to cosine similarity: similarity = 1 - (distance^2 / 2)
            # But for practical purposes, let's use the raw score and adjust threshold
            similarity = max(0.0, 1.0 - score)  # Ensure non-negative
            
            if similarity > similarity_threshold:
                filtered_docs.append((doc, similarity))
                print(f"   ✅ Chunk similarity: {similarity:.3f} | {doc.page_content[:80]}...")
            else:
                print(f"   ❌ Rejected similarity: {similarity:.3f} | {doc.page_content[:80]}...")
        
        # Sort by similarity (descending - highest first)
        filtered_docs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📈 Selected {len(filtered_docs)} high-quality chunks (similarity > {similarity_threshold})")
        
        return filtered_docs
    
    def build_structured_context(self, filtered_chunks: List[Tuple[Document, float]], max_chunks: int = 8) -> str:
        """Build structured context string with chunk headers - Bedrock KB style"""
        if not filtered_chunks:
            return ""
        
        context_parts = []
        chunks_to_use = filtered_chunks[:max_chunks]  # Take top 5-10 chunks
        
        print(f"🏗️ Building structured context from {len(chunks_to_use)} chunks")
        
        for i, (doc, similarity) in enumerate(chunks_to_use, 1):
            context_parts.append(f"### Chunk {i}:")
            context_parts.append(doc.page_content.strip())
            context_parts.append("")  # Empty line between chunks
        
        structured_context = "\n".join(context_parts)
        print(f"📝 Context length: {len(structured_context)} characters")
        
        return structured_context
    
    def maximal_marginal_relevance(self, query_embedding: np.ndarray, filtered_chunks: List[Tuple[Document, float]], 
                                 lambda_param: float = 0.5, k: int = 8) -> List[Tuple[Document, float]]:
        """Implement MMR for diverse chunk selection"""
        if not filtered_chunks or len(filtered_chunks) <= k:
            return filtered_chunks
        
        print(f"🎯 Applying MMR for diversity (λ={lambda_param}, k={k})")
        
        # Extract documents and their embeddings
        docs_with_scores = [(doc, score) for doc, score in filtered_chunks]
        doc_embeddings = []
        
        for doc, _ in docs_with_scores:
            embedding = self.bge_model.encode([doc.page_content])['dense_vecs'][0]
            doc_embeddings.append(embedding)
        
        doc_embeddings = np.array(doc_embeddings)
        
        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(docs_with_scores)))
        
        # Select first document (highest similarity)
        first_idx = 0  # Already sorted by similarity
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining documents balancing relevance and diversity
        while len(selected_indices) < k and remaining_indices:
            mmr_scores = []
            
            for i in remaining_indices:
                # Relevance score (similarity to query)
                relevance = docs_with_scores[i][1]  # Use pre-computed similarity
                
                # Diversity score (max similarity to already selected)
                if selected_indices:
                    similarities_to_selected = [
                        float(doc_embeddings[i].dot(doc_embeddings[j])) 
                        for j in selected_indices
                    ]
                    diversity = max(similarities_to_selected)
                else:
                    diversity = 0
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append((i, mmr_score))
            
            # Select document with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return selected documents with scores
        selected_docs = [docs_with_scores[i] for i in selected_indices]
        
        print(f"📊 MMR selected {len(selected_docs)} diverse chunks")
        for i, (doc, score) in enumerate(selected_docs, 1):
            print(f"   Chunk {i}: similarity={score:.3f} | {doc.page_content[:60]}...")
        
        return selected_docs
    
    def generate_answer_with_context(self, question: str, context: str) -> str:
        """Generate answer using custom prompt with structured context"""
        
        # Bedrock Knowledge Base style prompt
        bedrock_prompt = f"""You are a document analysis expert. Given the user question and supporting context from a meeting transcript, synthesize a clear, structured, and process-oriented answer using only the provided information.

Question: {question}

Context:
{context}

Answer:"""
        
        print("🧠 Generating answer with Bedrock KB style prompt...")
        
        try:
            response = self.claude_model.invoke(bedrock_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    def extract_relevant_clips(self, source_documents: List[Document], query: str) -> List[str]:
        """Extract video clips based on relevant source documents"""
        if not check_ffmpeg():
            print("⚠️ FFmpeg not available - cannot extract video clips")
            return []
        
        clips = []
        temp_dir = tempfile.mkdtemp()
        
        # Group documents by video file
        video_segments = {}
        for doc in source_documents:
            video_file = doc.metadata.get('video_file')
            if not video_file or not os.path.exists(video_file):
                continue
                
            if video_file not in video_segments:
                video_segments[video_file] = []
            
            start_time = doc.metadata.get('start_time')
            end_time = doc.metadata.get('end_time')
            
            if start_time and end_time:
                video_segments[video_file].append({
                    'start': time_str_to_seconds(start_time),
                    'end': time_str_to_seconds(end_time),
                    'text': doc.page_content
                })
        
        # Create clips for each video
        for video_file, segments in video_segments.items():
            if not segments:
                continue
            
            # Sort segments by start time
            segments.sort(key=lambda x: x['start'])
            
            # Merge overlapping or nearby segments (within 10 seconds)
            merged_segments = []
            current_segment = segments[0]
            
            for segment in segments[1:]:
                if segment['start'] - current_segment['end'] <= 10:  # Within 10 seconds
                    # Merge segments
                    current_segment['end'] = max(current_segment['end'], segment['end'])
                    current_segment['text'] += " " + segment['text']
                else:
                    merged_segments.append(current_segment)
                    current_segment = segment
            merged_segments.append(current_segment)
            
            # Create clips for merged segments
            for i, segment in enumerate(merged_segments):
                # Add buffer time (5 seconds before and after)
                start_with_buffer = max(0, segment['start'] - 5)
                end_with_buffer = segment['end'] + 5
                
                # Create output filename
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                clip_filename = f"{video_name}_clip_{i+1}_{int(start_with_buffer)}_{int(end_with_buffer)}.mp4"
                clip_path = os.path.join(temp_dir, clip_filename)
                
                # Extract clip
                if extract_video_clip(video_file, start_with_buffer, end_with_buffer, clip_path):
                    clips.append(clip_path)
                    print(f"✅ Created clip: {clip_filename} ({start_with_buffer:.1f}s - {end_with_buffer:.1f}s)")
                else:
                    print(f"❌ Failed to create clip: {clip_filename}")
        
        return clips
    
    def find_answer_supporting_clips(self, question: str, answer: str, max_clips: int = 2) -> Tuple[List[Document], List[Dict]]:
        """Find clips that best support the generated answer using answer-aware selection"""
        if not self.bge_model or not self.db:
            return [], []
        
        print("🎯 Finding clips that support the answer...")
        
        # Create a combined query from question + key parts of answer
        answer_sentences = answer.split('. ')[:2]  # Take first 2 sentences of answer
        combined_query = f"{question} {' '.join(answer_sentences)}"
        
        print(f"🔍 Combined search query: '{combined_query[:80]}...'")
        
        # Search for documents that match both question and answer content
        retriever = self.db.as_retriever(search_kwargs={"k": 8})  # Get more candidates
        candidate_docs = retriever.get_relevant_documents(combined_query)
        
        print(f"🔍 Evaluating {len(candidate_docs)} candidate chunks...")
        
        # Calculate similarity scores for each document
        scored_docs = []
        debug_info = []
        query_embedding = self.bge_model.encode([combined_query])['dense_vecs'][0]
        
        for i, doc in enumerate(candidate_docs, 1):
            doc_embedding = self.bge_model.encode([doc.page_content])['dense_vecs'][0]
            similarity = float(query_embedding.dot(doc_embedding))
            
            # Show chunk evaluation details
            chunk_preview = doc.page_content[:100].replace('\n', ' ') + "..."
            print(f"   Chunk {i}: Score={similarity:.3f} | {chunk_preview}")
            
            # Explain selection decision
            if similarity >= self.similarity_threshold:
                print(f"   ✅ SELECTED - High relevance to query")
                scored_docs.append({
                    'document': doc,
                    'similarity': similarity
                })
                
                # Extract key matching phrases for explanation
                key_words = [word for word in question.lower().split() if len(word) > 3]
                matching_words = [word for word in key_words if word in doc.page_content.lower()]
                
                debug_info.append({
                    'chunk_text': chunk_preview,
                    'similarity': similarity,
                    'start_time': doc.metadata.get('start_time', 'Unknown'),
                    'end_time': doc.metadata.get('end_time', 'Unknown'),
                    'matching_concepts': matching_words,
                    'selection_reason': f"Matches key concepts: {', '.join(matching_words[:3])}" if matching_words else "High semantic similarity"
                })
            else:
                print(f"   ❌ REJECTED - Below threshold ({self.similarity_threshold})")
        
        # Sort by similarity and take top clips
        scored_docs.sort(key=lambda x: x['similarity'], reverse=True)
        top_docs = [item['document'] for item in scored_docs[:max_clips]]
        top_debug = debug_info[:max_clips]
        
        print(f"📊 Selected {len(top_docs)} high-quality clips (similarity >= {self.similarity_threshold})")
        for i, item in enumerate(scored_docs[:max_clips], 1):
            print(f"   Clip {i}: similarity = {item['similarity']:.3f}")
        
        return top_docs, top_debug
    
    def query_with_clips(self, question: str) -> Dict:
        """Query the system and return both text answer and video clips with optimized selection"""
        print(f"\n🤔 Processing query: {question}")
        
        # Get answer from QA chain
        result = self.qa_chain.invoke({"query": question})
        answer = result['result']
        
        # Use answer-aware clip selection for better matching
        debug_info = []
        if 'source_documents' in result and result['source_documents']:
            print("🔍 Using answer-aware clip selection...")
            optimized_docs, debug_info = self.find_answer_supporting_clips(question, answer, max_clips=2)
            
            # If optimized selection found good clips, use those; otherwise fall back to original
            if optimized_docs:
                source_docs_for_clips = optimized_docs
                print(f"✅ Using {len(optimized_docs)} optimized clips")
            else:
                source_docs_for_clips = result['source_documents'][:3]  # Limit to top 3
                print("⚠️ Falling back to original document selection")
        else:
            source_docs_for_clips = result.get('source_documents', [])
        
        # Extract video clips from selected documents
        temp_clips = []
        permanent_clips = []
        
        if source_docs_for_clips:
            temp_clips = self.extract_relevant_clips(source_docs_for_clips, question)
            
            # Save clips permanently if clip manager is available
            if self.save_clips and self.clip_manager and temp_clips:
                print("💾 Saving clips permanently...")
                
                for temp_clip in temp_clips:
                    # Extract timing info from filename
                    filename = os.path.basename(temp_clip)
                    parts = filename.split('_')
                    
                    try:
                        start_time = float(parts[-2])
                        end_time = float(parts[-1].replace('.mp4', ''))
                        
                        # Find the corresponding video source
                        video_source = None
                        for doc in source_docs_for_clips:
                            if doc.metadata.get('video_file'):
                                video_source = doc.metadata['video_file']
                                break
                        
                        if video_source:
                            permanent_clip = self.clip_manager.save_clip_with_metadata(
                                temp_clip, question, answer, 
                                start_time, end_time, video_source
                            )
                            if permanent_clip:
                                permanent_clips.append(permanent_clip)
                    except (IndexError, ValueError) as e:
                        print(f"⚠️ Could not parse timing from {filename}: {e}")
                        permanent_clips.append(temp_clip)  # Keep temp clip if parsing fails
                
                # Clean up temporary clips
                if temp_clips:
                    self.clip_manager.cleanup_temp_clips(temp_clips)
            else:
                permanent_clips = temp_clips
        
        # Create clip timestamp information for UI display
        clip_timestamps = []
        for i, clip_path in enumerate(permanent_clips):
            filename = os.path.basename(clip_path)
            parts = filename.split('_')
            
            try:
                start_time = float(parts[-2])
                end_time = float(parts[-1].replace('.mp4', ''))
                duration = end_time - start_time
                
                # Get debug info if available
                debug_item = debug_info[i] if i < len(debug_info) else {}
                similarity = debug_item.get('similarity', 0.0)
                reason = debug_item.get('selection_reason', 'Selected by system')
                
                clip_timestamps.append({
                    'filename': filename,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'similarity_score': similarity,
                    'selection_reason': reason,
                    'start_time_formatted': f"{int(start_time//60)}:{int(start_time%60):02d}",
                    'end_time_formatted': f"{int(end_time//60)}:{int(end_time%60):02d}"
                })
            except (IndexError, ValueError):
                # Fallback for clips that couldn't be parsed
                clip_timestamps.append({
                    'filename': filename,
                    'start_time': 0,
                    'end_time': 0,
                    'duration': 0,
                    'similarity_score': 0.0,
                    'selection_reason': 'Timestamp parsing failed',
                    'start_time_formatted': '0:00',
                    'end_time_formatted': '0:00'
                })
        
        return {
            'question': question,
            'answer': answer,
            'video_clips': permanent_clips,
            'clip_timestamps': clip_timestamps,
            'debug_info': debug_info,
            'temp_clips': temp_clips,
            'source_documents': result.get('source_documents', []),
            'optimized_selection': len(permanent_clips) > 0
        }
    
    def bedrock_quality_query_with_clips(self, question: str) -> Dict:
        """Enhanced query method matching Bedrock KB quality with video clips"""
        print(f"\n🚀 Bedrock Quality Processing: {question}")
        
        # Step 1: Get filtered chunks (k=25, filter >0.68, sort by similarity)
        filtered_chunks = self.get_filtered_chunks(question, k=25, similarity_threshold=self.bedrock_similarity_threshold)
        
        if not filtered_chunks:
            print("❌ No chunks passed the similarity threshold")
            return {
                'question': question,
                'answer': "I couldn't find relevant information to answer your question.",
                'video_clips': [],
                'clip_timestamps': [],
                'debug_info': {'chunks_evaluated': 0, 'chunks_selected': 0, 'avg_similarity': 0.0},
                'bedrock_quality': True
            }
        
        # Step 2: Optional MMR for diversity
        if self.use_mmr and len(filtered_chunks) > 3:  # Apply MMR if we have more than 3 chunks
            query_embedding = self.bge_model.encode([question])['dense_vecs'][0]
            filtered_chunks = self.maximal_marginal_relevance(query_embedding, filtered_chunks, lambda_param=0.5, k=min(8, len(filtered_chunks)))
        
        # Step 3: Build structured context (top 5-10 chunks)
        context = self.build_structured_context(filtered_chunks, max_chunks=8)
        
        # Step 4: Generate answer with custom prompt
        answer = self.generate_answer_with_context(question, context)
        
        # Step 5: Extract clips from selected chunks (existing pipeline)
        source_docs = [doc for doc, score in filtered_chunks[:5]]  # Top 5 for clips
        temp_clips = []
        permanent_clips = []
        
        if source_docs:
            temp_clips = self.extract_relevant_clips(source_docs, question)
            
            # Save clips permanently if clip manager is available
            if self.save_clips and self.clip_manager and temp_clips:
                print("💾 Saving clips permanently...")
                
                for temp_clip in temp_clips:
                    # Extract timing info from filename
                    filename = os.path.basename(temp_clip)
                    parts = filename.split('_')
                    
                    try:
                        start_time = float(parts[-2])
                        end_time = float(parts[-1].replace('.mp4', ''))
                        
                        # Find the corresponding video source
                        video_source = None
                        for doc in source_docs:
                            if doc.metadata.get('video_file'):
                                video_source = doc.metadata['video_file']
                                break
                        
                        if video_source:
                            permanent_clip = self.clip_manager.save_clip_with_metadata(
                                temp_clip, question, answer, 
                                start_time, end_time, video_source
                            )
                            if permanent_clip:
                                permanent_clips.append(permanent_clip)
                    except (IndexError, ValueError) as e:
                        print(f"⚠️ Could not parse timing from {filename}: {e}")
                        permanent_clips.append(temp_clip)  # Keep temp clip if parsing fails
                
                # Clean up temporary clips
                if temp_clips:
                    self.clip_manager.cleanup_temp_clips(temp_clips)
            else:
                permanent_clips = temp_clips
        
        # Create enhanced clip timestamp information
        clip_timestamps = []
        for i, clip_path in enumerate(permanent_clips):
            filename = os.path.basename(clip_path)
            parts = filename.split('_')
            
            try:
                start_time = float(parts[-2])
                end_time = float(parts[-1].replace('.mp4', ''))
                duration = end_time - start_time
                
                # Get similarity from filtered chunks if available
                similarity = filtered_chunks[i][1] if i < len(filtered_chunks) else 0.0
                
                clip_timestamps.append({
                    'filename': filename,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'similarity_score': similarity,
                    'selection_reason': f"Bedrock Quality (similarity: {similarity:.3f})",
                    'start_time_formatted': f"{int(start_time//60)}:{int(start_time%60):02d}",
                    'end_time_formatted': f"{int(end_time//60)}:{int(end_time%60):02d}"
                })
            except (IndexError, ValueError):
                clip_timestamps.append({
                    'filename': filename,
                    'start_time': 0,
                    'end_time': 0,
                    'duration': 0,
                    'similarity_score': 0.0,
                    'selection_reason': 'Timestamp parsing failed',
                    'start_time_formatted': '0:00',
                    'end_time_formatted': '0:00'
                })
        
        # Calculate statistics
        avg_similarity = sum(score for _, score in filtered_chunks) / len(filtered_chunks) if filtered_chunks else 0.0
        
        return {
            'question': question,
            'answer': answer,
            'video_clips': permanent_clips,
            'clip_timestamps': clip_timestamps,
            'debug_info': {
                'chunks_evaluated': 25,
                'chunks_selected': len(filtered_chunks),
                'avg_similarity': avg_similarity,
                'similarity_threshold': self.bedrock_similarity_threshold,
                'mmr_applied': self.use_mmr and len(filtered_chunks) > 3
            },
            'temp_clips': temp_clips,
            'source_documents': source_docs,
            'bedrock_quality': True,
            'context_length': len(context)
        }

def main():
    if not check_ffmpeg():
        print("⚠️ FFmpeg not found. Video clip extraction will not work.")
        print("Please install FFmpeg to enable video clip functionality.")
    
    # Initialize the enhanced QA system
    qa_system = EnhancedQASystem()
    
    # Look for sentence files and corresponding videos
    sentence_files = [f for f in os.listdir('.') if f.endswith('_sentences.json')]
    
    if not sentence_files:
        print("❌ No timestamped sentence files found.")
        print("Please run timestampsentence.py first to create sentence data.")
        return
    
    print("📁 Available sentence files:")
    for i, f in enumerate(sentence_files, 1):
        print(f"{i}. {f}")
    
    # For now, use the first available file
    sentences_file = sentence_files[0]
    
    # Find corresponding video file
    base_name = sentences_file.replace('_sentences.json', '')
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    video_file = None
    
    for ext in video_extensions:
        potential_video = base_name + ext
        if os.path.exists(potential_video):
            video_file = potential_video
            break
    
    if not video_file:
        print(f"❌ No corresponding video file found for {sentences_file}")
        return
    
    print(f"🎥 Using video: {video_file}")
    print(f"📝 Using sentences: {sentences_file}")
    
    # Load timestamped content
    documents = qa_system.load_timestamped_content(sentences_file, video_file)
    print(f"📚 Created {len(documents)} timestamped document chunks")
    
    # Create vector store
    qa_system.create_vector_store(documents)
    
    # Setup QA chain
    qa_system.setup_qa_chain()
    
    print("\n🧠 Enhanced QA system ready! Ask questions and get video clips.")
    print("Type 'exit' to quit.\n")
    
    while True:
        question = input("Q: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        
        if not question:
            continue
        
        try:
            result = qa_system.query_with_clips(question)
            
            print(f"\nA: {result['answer']}")
            
            if result['video_clips']:
                print(f"\n🎬 Generated {len(result['video_clips'])} video clip(s):")
                for i, clip in enumerate(result['video_clips'], 1):
                    print(f"   {i}. {clip}")
                print("\nYou can play these clips with any video player.")
            else:
                print("\n📹 No video clips generated (no relevant timestamps found)")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")

if __name__ == "__main__":
    main()
