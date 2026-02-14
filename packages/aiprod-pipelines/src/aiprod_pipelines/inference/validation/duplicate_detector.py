"""Duplicate detection using perceptual hashing"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class DuplicateMatch:
    """Result of duplicate comparison"""
    file1: str
    file2: str
    similarity: float  # 0-1, where 1.0 = identical
    hash1: str
    hash2: str


class DuplicateDetector:
    """Detects duplicate or near-duplicate videos using perceptual hashing"""
    
    def __init__(self, hash_size: int = 16):
        """
        Initialize detector.
        
        Args:
            hash_size: Size of perceptual hash (16x16 = 256 bit hash)
        """
        self.hash_size = hash_size
    
    def get_perceptual_hash(self, frame_data: np.ndarray) -> str:
        """
        Compute perceptual hash of image frame.
        
        Args:
            frame_data: Image as numpy array (H, W, C)
            
        Returns:
            Perceptual hash as binary string
        """
        import cv2
        
        # Convert to grayscale
        if len(frame_data.shape) == 3:
            gray = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_data
        
        # Resize to hash_size x hash_size
        small = cv2.resize(gray, (self.hash_size, self.hash_size))
        
        # Compute average brightness
        avg = np.mean(small)
        
        # Create binary hash
        hash_bits = (small > avg).flatten()
        hash_str = ''.join(['1' if b else '0' for b in hash_bits])
        
        return hash_str
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hashes"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def hash_similarity(self, hash1: str, hash2: str) -> float:
        """
        Compute similarity between two hashes.
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Similarity score 0-1 (1.0 = identical)
        """
        if len(hash1) != len(hash2):
            return 0.0
        
        distance = self.hamming_distance(hash1, hash2)
        similarity = 1.0 - (distance / len(hash1))
        return similarity
    
    async def find_duplicates(
        self,
        video_files: List[str],
        threshold: float = 0.85,
        sample_only_first_frame: bool = True,
    ) -> List[DuplicateMatch]:
        """
        Find duplicate videos in list.
        
        Args:
            video_files: List of video file paths
            threshold: Similarity threshold for duplicates (0.85 = 85% match)
            sample_only_first_frame: If True, only hash first frame; else sample multiple frames
            
        Returns:
            List of DuplicateMatch results
        """
        import cv2
        from pathlib import Path
        
        # Extract hashes for all videos
        hashes = {}
        
        for video_path in video_files:
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue
                
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    hash_str = self.get_perceptual_hash(frame)
                    hashes[video_path] = hash_str
            
            except Exception as e:
                print(f"Error hashing {video_path}: {e}")
                continue
        
        # Find duplicate pairs
        duplicates = []
        compared = set()
        
        for i, (vid1, hash1) in enumerate(list(hashes.items())):
            for vid2, hash2 in list(hashes.items())[i+1:]:
                pair_key = (vid1, vid2) if vid1 < vid2 else (vid2, vid1)
                
                if pair_key not in compared:
                    compared.add(pair_key)
                    
                    similarity = self.hash_similarity(hash1, hash2)
                    
                    if similarity >= threshold:
                        duplicates.append(DuplicateMatch(
                            file1=vid1,
                            file2=vid2,
                            similarity=similarity,
                            hash1=hash1,
                            hash2=hash2,
                        ))
        
        return duplicates
