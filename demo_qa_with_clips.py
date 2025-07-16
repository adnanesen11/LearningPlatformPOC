#!/usr/bin/env python3
"""
Demo script for the Enhanced QA System with Video Clip Extraction

This script demonstrates how to use the enhanced QA system that can:
1. Answer questions based on video transcripts
2. Extract relevant video clips for each answer
3. Save clips permanently with metadata

Usage:
    python demo_qa_with_clips.py
"""

import os
import sys
from enhanced_qa_with_clips import EnhancedQASystem, check_ffmpeg
from clip_manager import VideoClipManager

def demo_questions():
    """Sample questions to demonstrate the system"""
    return [
        "What is the process for indexing documents?",
        "How do you identify different document types?",
        "What should you do if you can't figure out what a document is?",
        "How do you handle W2 documents?",
        "What is the underwriting transmittal summary?",
        "How do you deal with duplicate documents?",
        "What happens when you index documents?",
        "How do you know which documents go together?"
    ]

def run_demo():
    """Run the demo with sample questions"""
    print("🎬 Enhanced QA System with Video Clips - Demo")
    print("=" * 50)
    
    # Check prerequisites
    if not check_ffmpeg():
        print("⚠️ FFmpeg not found. Video clip extraction will not work.")
        print("Please install FFmpeg to enable video clip functionality.")
        print("Demo will continue with text-only responses.\n")
    
    # Initialize the enhanced QA system
    print("🚀 Initializing Enhanced QA System...")
    qa_system = EnhancedQASystem(save_clips=True)
    
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
    print("\n📚 Loading timestamped content...")
    documents = qa_system.load_timestamped_content(sentences_file, video_file)
    print(f"Created {len(documents)} timestamped document chunks")
    
    print("🔍 Creating vector store...")
    qa_system.create_vector_store(documents)
    
    print("🧠 Setting up QA chain...")
    qa_system.setup_qa_chain()
    
    print("\n✅ System ready!")
    print("=" * 50)
    
    # Run demo questions
    demo_qs = demo_questions()
    
    print(f"\n🎯 Running demo with {len(demo_qs)} sample questions:")
    print("(You can also ask your own questions)\n")
    
    for i, question in enumerate(demo_qs, 1):
        print(f"\n📝 Demo Question {i}/{len(demo_qs)}:")
        print(f"Q: {question}")
        
        try:
            result = qa_system.query_with_clips(question)
            
            print(f"\nA: {result['answer']}")
            
            if result['video_clips']:
                print(f"\n🎬 Generated {len(result['video_clips'])} video clip(s):")
                
                # Display detailed clip information with timestamps
                if 'clip_timestamps' in result and result['clip_timestamps']:
                    for j, clip_info in enumerate(result['clip_timestamps'], 1):
                        print(f"   {j}. {clip_info['filename']}")
                        print(f"      ⏱️  Time: {clip_info['start_time_formatted']} - {clip_info['end_time_formatted']} ({clip_info['duration']:.1f}s)")
                        print(f"      🎯 Relevance: {clip_info['similarity_score']:.3f}")
                        print(f"      📍 Why: {clip_info['selection_reason']}")
                        print()
                else:
                    # Fallback for clips without timestamp info
                    for j, clip in enumerate(result['video_clips'], 1):
                        clip_name = os.path.basename(clip)
                        print(f"   {j}. {clip_name}")
                
                print("💡 Clips saved permanently in 'generated_clips' folder")
            else:
                print("\n📹 No video clips generated for this question")
            
            print("\n" + "-" * 50)
            
            # Ask if user wants to continue
            if i < len(demo_qs):
                response = input("\nPress Enter to continue, 'q' to quit demo, or type your own question: ").strip()
                if response.lower() == 'q':
                    break
                elif response:
                    # User asked their own question
                    print(f"\n🤔 Your question: {response}")
                    try:
                        user_result = qa_system.query_with_clips(response)
                        print(f"\nA: {user_result['answer']}")
                        
                        if user_result['video_clips']:
                            print(f"\n🎬 Generated {len(user_result['video_clips'])} video clip(s):")
                            for j, clip in enumerate(user_result['video_clips'], 1):
                                clip_name = os.path.basename(clip)
                                print(f"   {j}. {clip_name}")
                        else:
                            print("\n📹 No video clips generated for this question")
                    except Exception as e:
                        print(f"❌ Error processing your question: {e}")
                    
                    print("\n" + "-" * 50)
            
        except Exception as e:
            print(f"❌ Error processing demo question: {e}")
            continue
    
    # Show clip statistics
    if qa_system.clip_manager:
        print("\n📊 Final Statistics:")
        stats = qa_system.clip_manager.get_clip_stats()
        print(f"   Total clips generated: {stats['total_clips']}")
        print(f"   Total storage used: {stats['total_size_mb']:.1f} MB")
        print(f"   Total video duration: {stats['total_duration']:.1f} seconds")
        
        if stats['total_clips'] > 0:
            print(f"\n💾 All clips saved in: {qa_system.clip_manager.clips_dir}")
            print("🔍 Use 'python clip_manager.py' to manage saved clips")
    
    print("\n🎉 Demo completed!")

def interactive_mode():
    """Run in interactive mode for custom questions"""
    print("🎬 Enhanced QA System - Interactive Mode")
    print("=" * 50)
    
    # Initialize system (same as demo)
    qa_system = EnhancedQASystem(save_clips=True)
    
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
    
    print(f"\n✅ System ready! Using {os.path.basename(video_file)}")
    print("Type 'exit' to quit, 'demo' to run demo questions\n")
    
    while True:
        question = input("Q: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            break
        elif question.lower() == 'demo':
            # Switch to demo mode
            run_demo()
            break
        elif not question:
            continue
        
        try:
            result = qa_system.query_with_clips(question)
            
            print(f"\nA: {result['answer']}")
            
            if result['video_clips']:
                print(f"\n🎬 Generated {len(result['video_clips'])} video clip(s):")
                
                # Display detailed clip information with timestamps
                if 'clip_timestamps' in result and result['clip_timestamps']:
                    for i, clip_info in enumerate(result['clip_timestamps'], 1):
                        print(f"   {i}. {clip_info['filename']}")
                        print(f"      ⏱️  Time: {clip_info['start_time_formatted']} - {clip_info['end_time_formatted']} ({clip_info['duration']:.1f}s)")
                        print(f"      🎯 Relevance: {clip_info['similarity_score']:.3f}")
                        print(f"      📍 Why: {clip_info['selection_reason']}")
                        print()
                else:
                    # Fallback for clips without timestamp info
                    for i, clip in enumerate(result['video_clips'], 1):
                        print(f"   {i}. {os.path.basename(clip)}")
            else:
                print("\n📹 No video clips generated")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

def bedrock_quality_mode():
    """Run in Bedrock Quality mode with enhanced retrieval"""
    print("🚀 Bedrock Quality Mode - Enhanced Retrieval")
    print("=" * 50)
    print("Features:")
    print("• k=25 candidate retrieval with similarity > 0.15 filtering")
    print("• Structured context with chunk headers")
    print("• Document analysis expert prompt")
    print("• Optional MMR for diversity")
    print()
    
    # Ask for MMR preference
    mmr_choice = input("Enable MMR (Maximal Marginal Relevance) for diversity? (y/n): ").strip().lower()
    use_mmr = mmr_choice in ['y', 'yes']
    
    # Initialize system with Bedrock Quality settings
    qa_system = EnhancedQASystem(
        save_clips=True, 
        use_bedrock_quality=True, 
        use_mmr=use_mmr,
        similarity_threshold=0.68
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
    print("📚 Loading timestamped content...")
    documents = qa_system.load_timestamped_content(sentences_file, video_file)
    print("🔍 Creating vector store...")
    qa_system.create_vector_store(documents)
    print("🧠 Setting up QA chain...")
    qa_system.setup_qa_chain()
    
    print(f"\n✅ Bedrock Quality System ready! Using {os.path.basename(video_file)}")
    print(f"🎯 MMR Diversity: {'Enabled' if use_mmr else 'Disabled'}")
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("Q: ").strip()
        
        if question.lower() in ['exit', 'quit']:
            break
        elif not question:
            continue
        
        try:
            # Use the enhanced Bedrock Quality method
            result = qa_system.bedrock_quality_query_with_clips(question)
            
            print(f"\nA: {result['answer']}")
            
            # Show enhanced debug info
            if 'debug_info' in result:
                debug = result['debug_info']
                print(f"\n📊 Bedrock Quality Stats:")
                print(f"   Chunks evaluated: {debug.get('chunks_evaluated', 0)}")
                print(f"   Chunks selected: {debug.get('chunks_selected', 0)}")
                print(f"   Avg similarity: {debug.get('avg_similarity', 0.0):.3f}")
                print(f"   Similarity threshold: {debug.get('similarity_threshold', 0.68)}")
                print(f"   MMR applied: {debug.get('mmr_applied', False)}")
                if 'context_length' in result:
                    print(f"   Context length: {result['context_length']} characters")
            
            if result['video_clips']:
                print(f"\n🎬 Generated {len(result['video_clips'])} video clip(s):")
                
                # Display detailed clip information with timestamps
                if 'clip_timestamps' in result and result['clip_timestamps']:
                    for i, clip_info in enumerate(result['clip_timestamps'], 1):
                        print(f"   {i}. {clip_info['filename']}")
                        print(f"      ⏱️  Time: {clip_info['start_time_formatted']} - {clip_info['end_time_formatted']} ({clip_info['duration']:.1f}s)")
                        print(f"      🎯 Relevance: {clip_info['similarity_score']:.3f}")
                        print(f"      📍 Why: {clip_info['selection_reason']}")
                        print()
                else:
                    # Fallback for clips without timestamp info
                    for i, clip in enumerate(result['video_clips'], 1):
                        print(f"   {i}. {os.path.basename(clip)}")
            else:
                print("\n📹 No video clips generated")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

def main():
    """Main entry point"""
    print("🎬 Enhanced QA System with Video Clips")
    print("=" * 50)
    print("Choose mode:")
    print("1. Interactive Mode (Ask any questions)")
    print("2. Demo Mode (Pre-defined questions)")
    print("3. Bedrock Quality Mode (Enhanced retrieval)")
    
    choice = input("\nSelect mode (1, 2, or 3): ").strip()
    
    if choice == '2':
        run_demo()
    elif choice == '3':
        bedrock_quality_mode()
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
