#!/usr/bin/env python3
"""
Demo script for the Enhanced QA System with BGE Reranker and Overlapping Chunks

This script demonstrates the improved RAG pipeline with:
1. Overlapping chunks for better context continuity
2. BGE-reranker-v2-m3 for superior relevance ranking
3. Enhanced retrieval pipeline with deduplication
4. Detailed performance metrics and debugging

Usage:
    python demo_enhanced_reranker.py
"""

import os
import sys
from enhanced_qa_with_reranker import EnhancedQASystemWithReranker, check_ffmpeg

def demo_questions():
    """Sample questions to demonstrate the enhanced system"""
    return [
        "What is the process for indexing documents?",
        "How do you identify different document types?",
        "What should you do if you can't figure out what a document is?",
        "How do you handle W2 documents?",
        "What is the underwriting transmittal summary?",
        "How do you deal with duplicate documents?",
        "What happens when you index documents?",
        "How do you know which documents go together?",
        "What should I always input before a loan number?",
        "What do I do if I can't figure out where the document fits?"
    ]

def run_enhanced_demo():
    """Run the demo with enhanced features"""
    print("🚀 Enhanced QA System with BGE Reranker - Demo")
    print("=" * 60)
    print("New Features:")
    print("  🔄 Overlapping chunks (20% overlap)")
    print("  🎯 BGE-reranker-v2-m3 for better relevance")
    print("  📊 Enhanced retrieval pipeline")
    print("  🔍 Intelligent deduplication")
    print("=" * 60)
    
    # Check prerequisites
    if not check_ffmpeg():
        print("⚠️ FFmpeg not found. Video clip extraction will not work.")
        print("Please install FFmpeg to enable video clip functionality.")
        print("Demo will continue with text-only responses.\n")
    
    # Initialize the enhanced QA system
    print("🚀 Initializing Enhanced QA System with Reranker...")
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
    
    print(f"📁 Found {len(sentence_files)} sentence file(s)")
    
    # Use the first available file
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
    
    print(f"🎥 Using video: {os.path.basename(video_file)}")
    print(f"📝 Using sentences: {os.path.basename(sentences_file)}")
    
    # Load and setup the system
    print("\n📚 Loading timestamped content with overlapping chunks...")
    documents = qa_system.load_timestamped_content(sentences_file, video_file)
    print(f"Created {len(documents)} timestamped document chunks")
    
    print("🔍 Creating vector store...")
    qa_system.create_vector_store(documents)
    
    print("🧠 Setting up QA chain...")
    qa_system.setup_qa_chain()
    
    print("\n✅ Enhanced System ready!")
    print("=" * 60)
    
    # Run demo questions
    demo_qs = demo_questions()
    
    print(f"\n🎯 Running demo with {len(demo_qs)} sample questions:")
    print("(You can also ask your own questions)\n")
    
    for i, question in enumerate(demo_qs, 1):
        print(f"\n📝 Demo Question {i}/{len(demo_qs)}:")
        print(f"Q: {question}")
        
        try:
            # Use the enhanced method with reranking
            result = qa_system.enhanced_query_with_reranking(question)
            
            print(f"\nA: {result['answer']}")
            
            # Show enhanced debug info
            if 'debug_info' in result:
                debug = result['debug_info']
                print(f"\n📊 Enhanced Retrieval Performance:")
                print(f"   🔍 Initial chunks retrieved: {debug.get('chunks_retrieved', 0)}")
                print(f"   🎯 After reranking & dedup: {debug.get('chunks_after_dedup', 0)}")
                print(f"   📈 Avg relevance score: {debug.get('avg_relevance_score', 0.0):.3f}")
                print(f"   🔄 Reranking applied: {debug.get('reranking_used', False)}")
                print(f"   📎 Overlapping chunks: {debug.get('overlapping_chunks_used', False)}")
                if 'context_length' in result:
                    print(f"   📝 Context length: {result['context_length']} chars")
            
            if result['video_clips']:
                print(f"\n🎬 Generated {len(result['video_clips'])} video clip(s):")
                
                # Display detailed clip information with relevance scores
                if 'clip_timestamps' in result and result['clip_timestamps']:
                    for j, clip_info in enumerate(result['clip_timestamps'], 1):
                        print(f"   {j}. {clip_info['filename']}")
                        print(f"      ⏱️  Time: {clip_info['start_time_formatted']} - {clip_info['end_time_formatted']} ({clip_info['duration']:.1f}s)")
                        print(f"      🎯 Relevance: {clip_info['relevance_score']:.3f}")
                        print(f"      📍 Selection: {clip_info['selection_reason']}")
                        print()
                else:
                    # Fallback for clips without timestamp info
                    for j, clip in enumerate(result['video_clips'], 1):
                        clip_name = os.path.basename(clip)
                        print(f"   {j}. {clip_name}")
                
                print("💾 Clips saved permanently in 'generated_clips' folder")
            else:
                print("\n📹 No video clips generated for this question")
            
            print("\n" + "-" * 60)
            
            # Ask if user wants to continue
            if i < len(demo_qs):
                response = input("\nPress Enter to continue, 'q' to quit demo, or type your own question: ").strip()
                if response.lower() == 'q':
                    break
                elif response:
                    # User asked their own question
                    print(f"\n🤔 Your question: {response}")
                    try:
                        user_result = qa_system.enhanced_query_with_reranking(response)
                        print(f"\nA: {user_result['answer']}")
                        
                        # Show performance stats for user question
                        if 'debug_info' in user_result:
                            debug = user_result['debug_info']
                            print(f"\n📊 Performance Stats:")
                            print(f"   Retrieved: {debug.get('chunks_retrieved', 0)} → {debug.get('chunks_after_dedup', 0)} chunks")
                            print(f"   Avg score: {debug.get('avg_relevance_score', 0.0):.3f}")
                        
                        if user_result['video_clips']:
                            print(f"\n🎬 Generated {len(user_result['video_clips'])} video clip(s):")
                            for j, clip in enumerate(user_result['video_clips'], 1):
                                clip_name = os.path.basename(clip)
                                print(f"   {j}. {clip_name}")
                        else:
                            print("\n📹 No video clips generated for this question")
                    except Exception as e:
                        print(f"❌ Error processing your question: {e}")
                    
                    print("\n" + "-" * 60)
            
        except Exception as e:
            print(f"❌ Error processing demo question: {e}")
            continue
    
    # Show final statistics
    if qa_system.clip_manager:
        print("\n📊 Final Performance Summary:")
        stats = qa_system.clip_manager.get_clip_stats()
        print(f"   🎬 Total clips generated: {stats['total_clips']}")
        print(f"   💾 Total storage used: {stats['total_size_mb']:.1f} MB")
        print(f"   ⏱️  Total video duration: {stats['total_duration']:.1f} seconds")
        print(f"   📈 Avg clip duration: {stats.get('avg_duration', 0):.1f} seconds")
        
        if stats['total_clips'] > 0:
            print(f"\n💾 All clips saved in: {qa_system.clip_manager.clips_dir}")
            print("🔍 Use 'python clip_manager.py' to manage saved clips")
    
    print("\n🎉 Enhanced Demo completed!")
    print("Key improvements demonstrated:")
    print("  ✅ Better chunk overlap for context continuity")
    print("  ✅ Superior relevance ranking with BGE reranker")
    print("  ✅ Intelligent deduplication of overlapping content")
    print("  ✅ Enhanced performance metrics and debugging")

def comparison_mode():
    """Run comparison between original and enhanced systems"""
    print("🔬 Comparison Mode: Original vs Enhanced System")
    print("=" * 60)
    
    # This would require importing both systems and running side-by-side
    # For now, just show the enhanced system with detailed metrics
    print("Enhanced system features:")
    print("  🔄 Overlapping chunks (20% overlap)")
    print("  🎯 BGE-reranker-v2-m3")
    print("  📊 Multi-stage retrieval pipeline")
    print("  🔍 Smart deduplication")
    print("  📈 Detailed performance metrics")
    print("\nRunning enhanced system...")
    
    run_enhanced_demo()

def interactive_enhanced_mode():
    """Run in interactive mode with enhanced features"""
    print("🎬 Enhanced QA System - Interactive Mode")
    print("=" * 60)
    
    # Initialize enhanced system
    qa_system = EnhancedQASystemWithReranker(
        save_clips=True,
        use_overlapping_chunks=True,
        chunk_overlap_ratio=0.2,
        use_reranker=True,
        rerank_top_k=10
    )
    
    sentence_files = [f for f in os.listdir('.') if f.endswith('_sentences.json')]
    if not sentence_files:
        print("❌ No timestamped sentence files found.")
        return
    
    sentences_file = sentence_files[0]
    base_name = sentences_file.replace('_sentences.json', '')
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    video_file = None
    
    for ext in video_extensions:
        potential_video = base_name + ext
        if os.path.exists(potential_video):
            video_file = potential_video
            break
    
    if not video_file:
        print(f"❌ No corresponding video file found.")
        return
    
    # Setup system
    documents = qa_system.load_timestamped_content(sentences_file, video_file)
    qa_system.create_vector_store(documents)
    qa_system.setup_qa_chain()
    
    print(f"\n✅ Enhanced system ready! Using {os.path.basename(video_file)}")
    print("Features: Overlapping chunks + BGE reranker + Smart deduplication")
    print("Type 'exit' to quit, 'demo' to run demo questions, 'stats' for performance info\n")
    
    while True:
        question = input("Q: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            break
        elif question.lower() == 'demo':
            # Switch to demo mode
            run_enhanced_demo()
            break
        elif question.lower() == 'stats':
            # Show system statistics
            if qa_system.clip_manager:
                stats = qa_system.clip_manager.get_clip_stats()
                print(f"\n📊 System Statistics:")
                print(f"   Total clips: {stats['total_clips']}")
                print(f"   Storage used: {stats['total_size_mb']:.1f} MB")
                print(f"   Total duration: {stats['total_duration']:.1f} seconds")
                print()
            continue
        elif not question:
            continue
        
        try:
            result = qa_system.enhanced_query_with_reranking(question)
            
            print(f"\nA: {result['answer']}")
            
            # Show performance metrics
            if 'debug_info' in result:
                debug = result['debug_info']
                print(f"\n📊 Performance:")
                print(f"   Retrieved: {debug.get('chunks_retrieved', 0)} → {debug.get('chunks_after_dedup', 0)} chunks")
                print(f"   Avg relevance: {debug.get('avg_relevance_score', 0.0):.3f}")
                print(f"   Reranking: {'✅' if debug.get('reranking_used', False) else '❌'}")
            
            if result['video_clips']:
                print(f"\n🎬 Generated {len(result['video_clips'])} video clip(s):")
                
                # Display detailed clip information with relevance scores
                if 'clip_timestamps' in result and result['clip_timestamps']:
                    for i, clip_info in enumerate(result['clip_timestamps'], 1):
                        print(f"   {i}. {clip_info['filename']}")
                        print(f"      ⏱️  {clip_info['start_time_formatted']}-{clip_info['end_time_formatted']} ({clip_info['duration']:.1f}s)")
                        print(f"      🎯 Score: {clip_info['relevance_score']:.3f}")
                        print()
                else:
                    for i, clip in enumerate(result['video_clips'], 1):
                        print(f"   {i}. {os.path.basename(clip)}")
            else:
                print("\n📹 No video clips generated")
            
            print("\n" + "="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

def main():
    """Main entry point"""
    print("🚀 Enhanced QA System with BGE Reranker Demo")
    print("=" * 60)
    print("Choose mode:")
    print("1. Interactive Mode (Ask any questions)")
    print("2. Demo Mode (Pre-defined questions)")
    print("3. Comparison Mode (Show improvements)")
    
    choice = input("\nSelect mode (1, 2, or 3): ").strip()
    
    if choice == '2':
        run_enhanced_demo()
    elif choice == '3':
        comparison_mode()
    else:
        interactive_enhanced_mode()

if __name__ == "__main__":
    main()
