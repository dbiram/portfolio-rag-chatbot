#!/usr/bin/env python3
"""
Example showing how experience.json prioritization works.

This script demonstrates the impact of source prioritization on search results.
"""

import asyncio
from app.core.config import settings
from app.services.retrieval import retrieval_service
from app.services.vector_store import vector_store


async def demo_prioritization():
    """Demonstrate experience prioritization with a side-by-side comparison."""
    
    # Load the vector store
    try:
        vector_store.load_index()
        print("✅ Vector store loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        print("Please run: python scripts/ingest.py")
        return
    
    print("\n" + "="*80)
    print("EXPERIENCE PRIORITIZATION DEMO")
    print("="*80)
    
    # Show current priority settings
    print(f"\n📋 Current Source Priority Settings:")
    for source, boost in settings.source_priority_boost.items():
        icon = "⭐" if boost > 1.0 else "📄"
        print(f"   {icon} {source}: {boost}x boost")
    
    print(f"\n🔍 Testing with work-related query...")
    query = "What experience do you have with data engineering and ETL?"
    print(f"Query: '{query}'")
    
    # Get results with prioritization
    results = await retrieval_service.retrieve_chunks(query, top_k=5)
    
    print(f"\n📊 Top 5 Results (with experience.json getting 1.3x boost):")
    print("-" * 60)
    
    experience_count = 0
    for i, (chunk, score) in enumerate(results, 1):
        source = chunk.get('source', 'unknown')
        title = chunk.get('title', 'Untitled')
        
        # Calculate original score before boost
        boost = settings.source_priority_boost.get(source, 1.0)
        original_score = score / boost
        
        # Mark experience chunks
        icon = "⭐" if source == 'experience.json' else "📄"
        if source == 'experience.json':
            experience_count += 1
        
        print(f"{i}. {icon} {title}")
        print(f"   Source: {source}")
        print(f"   Original: {original_score:.3f} → Boosted: {score:.3f} (×{boost})")
        print(f"   Text preview: {chunk.get('text', '')[:100]}...")
        print()
    
    # Summary
    print("📈 RESULTS SUMMARY:")
    print(f"   • Experience chunks in top 5: {experience_count}/5")
    print(f"   • Experience content is {'✅ prioritized' if experience_count >= 2 else '❌ not well prioritized'}")
    
    if experience_count >= 2:
        print("   • ✅ The prioritization is working! Experience content appears at the top.")
    else:
        print("   • ⚠️  Consider increasing the boost factor for experience.json")
    
    print("\n" + "="*80)
    print("💡 HOW TO ADJUST PRIORITIZATION:")
    print("="*80)
    print("Edit the source_priority_boost values in app/core/config.py:")
    print("- Values > 1.0 increase priority (e.g., 1.3 = 30% boost)")
    print("- Values < 1.0 decrease priority (e.g., 0.9 = 10% reduction)")
    print("- Value = 1.0 means no change to original similarity score")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(demo_prioritization())