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
from FlagEmbedding import BGEM3FlagModel, FlagReranker
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

class BGE_Reranker:
    def __init__(self):
        print("🚀 Loading BGE Reranker v2-m3...")
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        print("✅ BGE Reranker loaded successfully")

    def rerank(self, query: str, passages: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """
        Rerank passages based on query relevance
        Returns: List of (original_index, score) tuples sorted by relevance
        """
        if not passages:
            return []
        
        # Prepare query-passage pairs for reranking
        pairs = [[query, passage] for passage in passages]
        
        print(f"🔄 Reranking {len(passages)} passages...")
        scores = self.reranker.compute_score(pairs, normalize=True)
        
        # Handle single passage case
        if isinstance(scores, float):
            scores = [scores]
        
        # Create (index, score) pairs and sort by score (descending)
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k:
            indexed_scores = indexed_scores[:top_k]
        
        print(f"📊 Reranking complete. Top score: {indexed_scores[0][1]:.3f}")
        return indexed_scores

bge_m3_embedder = BGE_M3_Embedder()
bge_reranker = BGE_Reranker()

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

class EnhancedQASystemWithReranker:
    def __init__(self, save_clips: bool = True, similarity_threshold: float = 0.65, 
                 use_overlapping_chunks: bool = True, chunk_overlap_ratio: float = 0.2,
                 use_reranker: bool = True, rerank_top_k: int = 10):
        self.db = None
        self.qa_chain = None
        self.video_files = {}  # Map video names to file paths
        self.sentence_data = {}  # Map video names to sentence data
        self.save_clips = save_clips
        self.clip_manager = VideoClipManager() if save_clips else None
        self.similarity_threshold = similarity_threshold
        self.bge_model = None  # For embeddings
        self.use_overlapping_chunks = use_overlapping_chunks
        self.chunk_overlap_ratio = chunk_overlap_ratio
        self.use_reranker = use_reranker
        self.rerank_top_k = rerank_top_k
        self.claude_model = None
        
    def create_overlapping_chunks(self, sentences: List[Dict], chunk_size: int = 4) -> List[Dict]:
        """Create overlapping chunks from sentences"""
        if not self.use_overlapping_chunks:
            # Original non-overlapping approach
            chunks = []
            for i in range(0, len(sentences), chunk_size):
                chunk_sentences = sentences[i:i + chunk_size]
                chunks.append({
                    'sentences': chunk_sentences,
                    'start_index': i,
                    'end_index': min(i + chunk_size - 1, len(sentences) - 1)
                })
            return chunks
        
        # New overlapping approach
        chunks = []
        overlap_size = max(1, int(chunk_size * self.chunk_overlap_ratio))
        step_size = chunk_size - overlap_size
        
        print(f"📊 Creating overlapping chunks: size={chunk_size}, overlap={overlap_size}, step={step_size}")
        
        i = 0
        while i < len(sentences):
            end_idx = min(i + chunk_size, len(sentences))
            chunk_sentences = sentences[i:end_idx]
            
            # Only create chunk if it has meaningful content
            if len(chunk_sentences) >= max(1, chunk_size // 2):
                chunks.append({
                    'sentences': chunk_sentences,
                    'start_index': i,
                    'end_index': end_idx - 1,
                    'is_overlapping': i > 0  # Mark overlapping chunks
                })
            
            # Move by step_size for overlapping, or break if we're at the end
            if end_idx >= len(sentences):
                break
            i += step_size
        
        print(f"✅ Created {len(chunks)} chunks ({sum(1 for c in chunks if c.get('is_overlapping', False))} overlapping)")
        return chunks
        
    def load_timestamped_content(self, sentences_file: str, video_file: str):
        """Load timestamped sentences and create enhanced documents with overlapping chunks"""
        print(f"📚 Loading timestamped content from {sentences_file}")
        
        # Load sentence data
        with open(sentences_file, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        
        # Store for later use
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        self.video_files[video_name] = video_file
        self.sentence_data[video_name] = sentences
        
        # Create overlapping chunks
        chunks = self.create_overlapping_chunks(sentences, chunk_size=4)
        
        # Create documents with timestamp metadata
        documents = []
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_sentences = chunk['sentences']
            
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
                    'chunk_index': chunk_idx,
                    'sentence_count': len(chunk_sentences),
                    'video_name': video_name,
                    'start_sentence_idx': chunk['start_index'],
                    'end_sentence_idx': chunk['end_index'],
                    'is_overlapping': chunk.get('is_overlapping', False)
                }
            )
            documents.append(doc)
        
        print(f"📊 Created {len(documents)} timestamped document chunks")
        if self.use_overlapping_chunks:
            overlapping_count = sum(1 for doc in documents if doc.metadata.get('is_overlapping', False))
            print(f"   📎 {overlapping_count} chunks have overlapping content")
        
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
        retriever = self.db.as_retriever(search_kwargs={"k": 15})  # Get more candidates for reranking
        self.claude_model = ChatBedrock(
            client=bedrock_client,
            model_id="arn:aws:bedrock:us-east-1:225989333617:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            provider="anthropic"
        )
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.claude_model, retriever=retriever, return_source_documents=True)
    
    def enhanced_retrieval_with_reranking(self, query: str, k: int = 25) -> List[Tuple[Document, float]]:
        """Enhanced retrieval with reranking pipeline"""
        if not self.db:
            return []
        
        print(f"🔍 Enhanced Retrieval Pipeline for: '{query[:60]}...'")
        
        # Step 1: Initial broad retrieval
        print(f"📥 Step 1: Retrieving {k} candidate chunks...")
        candidate_docs = self.db.similarity_search_with_score(query, k=k)
        
        if not candidate_docs:
            print("❌ No candidates found")
            return []
        
        print(f"✅ Retrieved {len(candidate_docs)} candidates")
        
        # Step 2: Apply reranking if enabled
        if self.use_reranker and len(candidate_docs) > 1:
            print(f"🔄 Step 2: Reranking with BGE-reranker-v2-m3...")
            
            # Extract passages and documents
            passages = [doc.page_content for doc, _ in candidate_docs]
            docs = [doc for doc, _ in candidate_docs]
            
            # Rerank passages
            reranked_results = bge_reranker.rerank(query, passages, top_k=self.rerank_top_k)
            
            # Reconstruct results with reranker scores
            reranked_docs = []
            for original_idx, rerank_score in reranked_results:
                doc = docs[original_idx]
                reranked_docs.append((doc, rerank_score))
                print(f"   📊 Chunk {original_idx}: rerank_score={rerank_score:.3f} | {doc.page_content[:60]}...")
            
            print(f"✅ Reranking complete. Selected top {len(reranked_docs)} chunks")
            return reranked_docs
        else:
            print("⏭️ Step 2: Skipping reranking (disabled or insufficient candidates)")
            # Convert similarity distances to scores (higher is better)
            scored_docs = []
            for doc, distance in candidate_docs:
                # Convert L2 distance to similarity score
                similarity_score = max(0.0, 1.0 - distance)
                scored_docs.append((doc, similarity_score))
            
            return scored_docs[:self.rerank_top_k]
    
    def deduplicate_overlapping_chunks(self, docs_with_scores: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Remove redundant overlapping chunks while preserving the best ones"""
        if not self.use_overlapping_chunks:
            return docs_with_scores
        
        print("🔄 Deduplicating overlapping chunks...")
        
        # Group chunks by video and time proximity
        video_groups = {}
        for doc, score in docs_with_scores:
            video_name = doc.metadata.get('video_name', 'unknown')
            start_time = time_str_to_seconds(doc.metadata.get('start_time', '0:00:00'))
            
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append((doc, score, start_time))
        
        # Deduplicate within each video
        deduplicated = []
        for video_name, chunks in video_groups.items():
            # Sort by start time
            chunks.sort(key=lambda x: x[2])  # Sort by start_time
            
            selected_chunks = []
            for doc, score, start_time in chunks:
                # Check if this chunk significantly overlaps with already selected chunks
                is_redundant = False
                for selected_doc, selected_score, selected_start_time in selected_chunks:
                    time_diff = abs(start_time - selected_start_time)
                    
                    # If chunks are very close in time (within 30 seconds), consider overlap
                    if time_diff < 30:
                        # Keep the one with higher score
                        if score <= selected_score:
                            is_redundant = True
                            break
                        else:
                            # Remove the lower-scoring chunk
                            selected_chunks = [(d, s, t) for d, s, t in selected_chunks 
                                             if not (abs(t - selected_start_time) < 30)]
                
                if not is_redundant:
                    selected_chunks.append((doc, score, start_time))
            
            # Add to final results
            for doc, score, _ in selected_chunks:
                deduplicated.append((doc, score))
        
        # Sort by score (descending)
        deduplicated.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ Deduplication complete: {len(docs_with_scores)} → {len(deduplicated)} chunks")
        return deduplicated
    
    def build_structured_context(self, filtered_chunks: List[Tuple[Document, float]], max_chunks: int = 8) -> str:
        """Build structured context string with chunk headers"""
        if not filtered_chunks:
            return ""
        
        context_parts = []
        chunks_to_use = filtered_chunks[:max_chunks]
        
        print(f"🏗️ Building structured context from {len(chunks_to_use)} chunks")
        
        for i, (doc, score) in enumerate(chunks_to_use, 1):
            context_parts.append(f"### Chunk {i} (Score: {score:.3f}):")
            context_parts.append(doc.page_content.strip())
            context_parts.append("")  # Empty line between chunks
        
        structured_context = "\n".join(context_parts)
        print(f"📝 Context length: {len(structured_context)} characters")
        
        return structured_context
    
    def generate_answer_with_context(self, question: str, context: str) -> str:
        """Generate answer using custom prompt with structured context"""
        
        # Enhanced prompt for better answer quality
        enhanced_prompt = f"""You are a precise meeting transcript analyst. Extract and synthesize information from the provided context to answer the user's question directly and completely.

RULES:
1. Answer the question using ONLY the provided context
2. Be specific and detailed - extract all relevant facts
3. Structure your response clearly with bullet points or sections
4. If you find relevant information, state it confidently
5. Only say "insufficient information" if the context truly lacks relevant details

Question: {question}

Context:
{context}

Provide a complete, structured answer based on the context above:"""
        
        print("🧠 Generating enhanced answer with reranked context...")
        
        try:
            response = self.claude_model.invoke(enhanced_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer."
    
    def enhanced_query_with_reranking(self, question: str) -> Dict:
        """Enhanced query method with reranking and overlapping chunks"""
        print(f"\n🚀 Enhanced Query Processing: {question}")
        
        # Step 1: Enhanced retrieval with reranking
        retrieved_chunks = self.enhanced_retrieval_with_reranking(question, k=25)
        
        if not retrieved_chunks:
            print("❌ No relevant chunks found")
            return {
                'question': question,
                'answer': "I couldn't find relevant information to answer your question.",
                'video_clips': [],
                'clip_timestamps': [],
                'debug_info': {'chunks_retrieved': 0, 'chunks_after_dedup': 0, 'reranking_used': self.use_reranker},
                'enhanced_retrieval': True
            }
        
        # Step 2: Deduplicate overlapping chunks
        deduplicated_chunks = self.deduplicate_overlapping_chunks(retrieved_chunks)
        
        # Step 3: Build structured context
        context = self.build_structured_context(deduplicated_chunks, max_chunks=8)
        
        # Step 4: Generate enhanced answer
        answer = self.generate_answer_with_context(question, context)
        
        # Step 5: Extract clips from top chunks
        source_docs = [doc for doc, score in deduplicated_chunks[:5]]  # Top 5 for clips
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
                        permanent_clips.append(temp_clip)
                
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
                
                # Get score from deduplicated chunks if available
                score = deduplicated_chunks[i][1] if i < len(deduplicated_chunks) else 0.0
                
                clip_timestamps.append({
                    'filename': filename,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'relevance_score': score,
                    'selection_reason': f"Enhanced retrieval (score: {score:.3f})",
                    'start_time_formatted': f"{int(start_time//60)}:{int(start_time%60):02d}",
                    'end_time_formatted': f"{int(end_time//60)}:{int(end_time%60):02d}"
                })
            except (IndexError, ValueError):
                clip_timestamps.append({
                    'filename': filename,
                    'start_time': 0,
                    'end_time': 0,
                    'duration': 0,
                    'relevance_score': 0.0,
                    'selection_reason': 'Timestamp parsing failed',
                    'start_time_formatted': '0:00',
                    'end_time_formatted': '0:00'
                })
        
        # Calculate statistics
        avg_score = sum(score for _, score in deduplicated_chunks) / len(deduplicated_chunks) if deduplicated_chunks else 0.0
        
        return {
            'question': question,
            'answer': answer,
            'video_clips': permanent_clips,
            'clip_timestamps': clip_timestamps,
            'debug_info': {
                'chunks_retrieved': len(retrieved_chunks),
                'chunks_after_dedup': len(deduplicated_chunks),
                'avg_relevance_score': avg_score,
                'reranking_used': self.use_reranker,
                'overlapping_chunks_used': self.use_overlapping_chunks,
                'overlap_ratio': self.chunk_overlap_ratio
            },
            'temp_clips': temp_clips,
            'source_documents': source_docs,
            'enhanced_retrieval': True,
            'context_length': len(context)
        }
    
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

def main():
    if not check_ffmpeg():
        print("⚠️ FFmpeg not found. Video clip extraction will not work.")
        print("Please install FFmpeg to enable video clip functionality.")
    
    print("🚀 Enhanced QA System with Reranking and Overlapping Chunks")
    print("=" * 60)
    
    # Initialize the enhanced QA system
    qa_system = EnhancedQASystemWithReranker(
        save_clips=True,
        use_overlapping_chunks=True,
        chunk_overlap_ratio=0.2,  # 20% overlap
        use_reranker=True,
        rerank_top_k=10
    )
    
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
    
    # Load timestamped content with overlapping chunks
    documents = qa_system.load_timestamped_content(sentences_file, video_file)
    print(f"📚 Created {len(documents)} timestamped document chunks")
    
    # Create vector store
    qa_system.create_vector_store(documents)
    
    # Setup QA chain
    qa_system.setup_qa_chain()
    
    print("\n🧠 Enhanced QA system with reranking ready!")
    print("Features enabled:")
    print(f"  🔄 Overlapping chunks: {qa_system.use_overlapping_chunks} (overlap: {qa_system.chunk_overlap_ratio*100:.0f}%)")
    print(f"  🎯 BGE Reranker: {qa_system.use_reranker} (top-k: {qa_system.rerank_top_k})")
    print("Type 'exit' to quit.\n")
    
    while True:
        question = input("Q: ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        
        if not question:
            continue
        
        try:
            result = qa_system.enhanced_query_with_reranking(question)
            
            print(f"\nA: {result['answer']}")
            
            # Show enhanced debug info
            if 'debug_info' in result:
                debug = result['debug_info']
                print(f"\n📊 Enhanced Retrieval Stats:")
                print(f"   Chunks retrieved: {debug.get('chunks_retrieved', 0)}")
                print(f"   After deduplication: {debug.get('chunks_after_dedup', 0)}")
                print(f"   Avg relevance score: {debug.get('avg_relevance_score', 0.0):.3f}")
                print(f"   Reranking used: {debug.get('reranking_used', False)}")
                print(f"   Overlapping chunks: {debug.get('overlapping_chunks_used', False)}")
                if 'context_length' in result:
                    print(f"   Context length: {result['context_length']} characters")
            
            if result['video_clips']:
                print(f"\n🎬 Generated {len(result['video_clips'])} video clip(s):")
                
                # Display detailed clip information with relevance scores
                if 'clip_timestamps' in result and result['clip_timestamps']:
                    for i, clip_info in enumerate(result['clip_timestamps'], 1):
                        print(f"   {i}. {clip_info['filename']}")
                        print(f"      ⏱️  Time: {clip_info['start_time_formatted']} - {clip_info['end_time_formatted']} ({clip_info['duration']:.1f}s)")
                        print(f"      🎯 Relevance: {clip_info['relevance_score']:.3f}")
                        print(f"      📍 Why: {clip_info['selection_reason']}")
                        print()
                else:
                    # Fallback for clips without timestamp info
                    for i, clip in enumerate(result['video_clips'], 1):
                        print(f"   {i}. {os.path.basename(clip)}")
                
                print("💡 Clips saved permanently in 'generated_clips' folder")
            else:
                print("\n📹 No video clips generated")
            
            print("\n" + "="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")

if __name__ == "__main__":
    main()
