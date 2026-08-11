#!/usr/bin/env python3
"""
Import diseases from CSV file to database
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from app.config import settings
from app.database.database import init_db, get_session
from app.services.disease_service import DiseaseService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Import diseases from CSV."""
    csv_path = "data/diseases.csv"
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV file not found: {csv_path}")
        print(f"\nUsage: python scripts/import_diseases.py [path_to_csv]")
        print(f"Default: data/diseases.csv")
        sys.exit(1)
    
    # Initialize database
    logger.info("🚀 Initializing database...")
    init_db()
    
    # Import diseases
    logger.info(f"📥 Importing diseases from {csv_path}...")
    
    session = get_session()
    results = DiseaseService.import_from_csv(session, csv_path)
    session.close()
    
    # Print results
    print("\n" + "="*60)
    print("IMPORT RESULTS")
    print("="*60)
    print(f"Total rows:     {results['total']}")
    print(f"Created:        {results['created']}")
    print(f"Updated:        {results['updated']}")
    print(f"Errors:         {results['errors']}")
    print("="*60)
    
    if results['details']:
        print("\nDetails:")
        for detail in results['details'][:20]:  # Show first 20
            print(f"  - {detail}")
        
        if len(results['details']) > 20:
            print(f"  ... and {len(results['details']) - 20} more")
    
    print("="*60)
    
    if results['errors'] > 0:
        logger.warning(f"⚠️ Import completed with {results['errors']} errors")
        sys.exit(1)
    else:
        logger.info(f"✅ Import completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
