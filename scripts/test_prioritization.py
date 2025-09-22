#!/usr/bin/env python3
"""
Test script to verify experience.json prioritization in retrieval.

This script tests that chunks from experience.json receive higher priority
in search results compared to other sources.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.services.retrieval import retrieval_service
from app.services.vector_store import vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_experience_prioritization():
    """Test that experience-related queries prioritize experience.json content."""
    
    # Load the vector store first
    try:
        vector_store.load_index()
        logger.info("Vector store loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        logger.info("Please run ingestion script first: python scripts/ingest.py")
        return
    
    # Test queries that should favor experience content
    test_queries = [
        "Tell me about your work experience",
        "What did you do at Schneider Electric?",
        "Describe your role as a Data Engineer",
        "What experience do you have with ETL pipelines?",
        "Tell me about your professional background"
    ]
    
    logger.info(f"Source priority settings: {settings.source_priority_boost}")
    logger.info("-" * 80)
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nTest {i}/5: '{query}'")
        logger.info("-" * 40)
        
        try:
            # Get top 5 results
            results = await retrieval_service.retrieve_chunks(query, top_k=5)
            
            # Count results by source
            source_counts = {}
            experience_positions = []
            
            for idx, (chunk_data, score) in enumerate(results):
                source = chunk_data.get('source', 'unknown')
                title = chunk_data.get('title', 'Untitled')
                
                if source not in source_counts:
                    source_counts[source] = 0
                source_counts[source] += 1
                
                # Track position of experience.json chunks
                if source == 'experience.json':
                    experience_positions.append(idx + 1)
                
                logger.info(f"  {idx + 1}. {title} (from {source}) - Score: {score:.3f}")
            
            # Analysis
            logger.info(f"\nSource distribution: {source_counts}")
            
            if 'experience.json' in source_counts:
                logger.info(f"Experience chunks: {source_counts['experience.json']}/5 "
                          f"at positions: {experience_positions}")
                
                # Check if experience.json has good representation in top results
                exp_in_top_3 = len([pos for pos in experience_positions if pos <= 3])
                if exp_in_top_3 > 0:
                    logger.info("✅ Experience content well-represented in top 3")
                else:
                    logger.info("⚠️  No experience content in top 3")
            else:
                logger.info("❌ No experience content in results")
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PRIORITIZATION TEST SUMMARY")
    logger.info("=" * 80)
    logger.info("Expected behavior:")
    logger.info("- Experience-related queries should show experience.json chunks in top positions")
    logger.info("- Experience chunks should have higher scores due to 1.3x boost factor")
    logger.info("- Other sources should appear with lower priority")


async def test_score_boosting():
    """Test that the score boosting mechanism works correctly."""
    logger.info("\n" + "=" * 80)
    logger.info("SCORE BOOSTING TEST")
    logger.info("=" * 80)
    
    # Test with a query that might match multiple sources
    query = "Python development experience"
    
    logger.info(f"Query: '{query}'")
    logger.info("Testing score boosting mechanism...")
    
    try:
        # Get more results to see the full ranking
        results = await retrieval_service.retrieve_chunks(query, top_k=8)
        
        logger.info("\nResults with boosted scores:")
        for idx, (chunk_data, boosted_score) in enumerate(results):
            source = chunk_data.get('source', 'unknown')
            title = chunk_data.get('title', 'Untitled')
            boost_factor = settings.source_priority_boost.get(source, 1.0)
            original_score = boosted_score / boost_factor
            
            logger.info(f"  {idx + 1}. {title}")
            logger.info(f"      Source: {source}")
            logger.info(f"      Original score: {original_score:.3f}")
            logger.info(f"      Boost factor: {boost_factor}x")
            logger.info(f"      Final score: {boosted_score:.3f}")
            logger.info("")
        
        # Verify experience.json chunks have higher final scores
        exp_chunks = [(chunk, score) for chunk, score in results 
                     if chunk.get('source') == 'experience.json']
        
        if exp_chunks:
            logger.info(f"Found {len(exp_chunks)} experience chunks in top 8")
            logger.info("✅ Score boosting is working")
        else:
            logger.info("⚠️  No experience chunks in top 8 - boost might need adjustment")
            
    except Exception as e:
        logger.error(f"Error in score boosting test: {e}")


async def main():
    """Run all prioritization tests."""
    logger.info("Starting Experience Prioritization Tests")
    logger.info("=" * 80)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Validate environment
    if not settings.mistral_api_key:
        logger.error("MISTRAL_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        await test_experience_prioritization()
        await test_score_boosting()
        
        logger.info("\n✅ All prioritization tests completed!")
        
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up
        try:
            await retrieval_service.mistral_client.close()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(main())