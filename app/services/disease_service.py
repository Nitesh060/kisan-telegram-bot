"""
Disease service for business logic
"""

import logging
from typing import Optional, List
from sqlalchemy.orm import Session
from app.database.repository import DiseaseRepository
from app.database.models import Disease

logger = logging.getLogger(__name__)


class DiseaseService:
    """Business logic for disease operations."""
    
    @staticmethod
    def get_disease_info(
        session: Session,
        crop: str,
        disease: str,
    ) -> Optional[Disease]:
        """Get disease information."""
        return DiseaseRepository.get_disease(session, crop, disease)
    
    @staticmethod
    def search_by_keywords(session: Session, keywords: List[str]) -> List[Disease]:
        """Search diseases by keywords."""
        all_diseases = set()
        
        for keyword in keywords:
            diseases = DiseaseRepository.search_diseases(session, keyword)
            all_diseases.update(diseases)
        
        return list(all_diseases)
    
    @staticmethod
    def get_crop_specific_diseases(session: Session, crop: str) -> List[Disease]:
        """Get all diseases for a specific crop."""
        return DiseaseRepository.get_diseases_by_crop(session, crop)
    
    @staticmethod
    def import_from_csv(session: Session, csv_path: str, skip_duplicates: bool = True) -> dict:
        """
        Import diseases from CSV file.
        
        Expected CSV columns:
        - crop
        - disease_name
        - disease_type
        - scientific_name
        - symptoms
        - causes
        - favorable_conditions
        - management
        - prevention
        - treatment
        - severity
        - keywords
        """
        import csv
        
        results = {
            "total": 0,
            "created": 0,
            "updated": 0,
            "errors": 0,
            "details": [],
        }
        
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                for idx, row in enumerate(reader, 1):
                    results["total"] += 1
                    
                    try:
                        crop = row.get("crop", "").strip()
                        disease_name = row.get("disease_name", "").strip()
                        
                        if not crop or not disease_name:
                            results["errors"] += 1
                            results["details"].append(f"Row {idx}: Missing crop or disease_name")
                            continue
                        
                        # Check if exists
                        exists = DiseaseRepository.disease_exists(session, crop, disease_name)
                        
                        if exists and skip_duplicates:
                            results["details"].append(f"Row {idx}: Skipped (duplicate)")
                            continue
                        
                        # Create or update
                        disease = DiseaseRepository.create_or_update_disease(
                            session,
                            crop=crop,
                            disease_name=disease_name,
                            disease_type=row.get("disease_type", "Unknown"),
                            symptoms=row.get("symptoms", ""),
                            management=row.get("management", ""),
                            scientific_name=row.get("scientific_name"),
                            causes=row.get("causes"),
                            favorable_conditions=row.get("favorable_conditions"),
                            prevention=row.get("prevention"),
                            treatment=row.get("treatment"),
                            severity=row.get("severity", "Unknown"),
                            keywords=row.get("keywords"),
                        )
                        
                        if exists:
                            results["updated"] += 1
                            results["details"].append(f"Row {idx}: Updated {crop} - {disease_name}")
                        else:
                            results["created"] += 1
                            results["details"].append(f"Row {idx}: Created {crop} - {disease_name}")
                    
                    except Exception as e:
                        results["errors"] += 1
                        results["details"].append(f"Row {idx}: Error - {str(e)}")
                        logger.error(f"Error importing row {idx}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            results["errors"] += 1
            results["details"].append(f"File error: {str(e)}")
        
        return results
