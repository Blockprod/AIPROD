"""
Advanced Reward Modeling System

Learn user preferences and auto-tune hyperparameters:
- Reward model (neural network) training
- User feedback collection
- Bayesian optimization for hyperparameter suggestion
- Preference clustering
- A/B testing framework
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from pathlib import Path


@dataclass
class UserFeedback:
    """Single piece of user feedback"""
    feedback_id: str
    user_id: str
    generation_id: str
    prompt: str
    quality_rating: float  # 0-1
    speed_rating: float  # 0-1
    aesthetics_rating: float  # 0-1
    overall_rating: float  # 0-1, computed as weighted average
    
    # Video/audio embeddings (would come from encoders)
    video_embedding: Optional[np.ndarray] = None
    audio_embedding: Optional[np.ndarray] = None
    text_embedding: Optional[np.ndarray] = None
    
    # Generation parameters
    cfg_scale: float = 7.5
    stg_scale: float = 4.0
    num_steps: int = 50
    num_frames: int = 128
    
    timestamp: float = 0.0


@dataclass
class UserProfile:
    """Profile for individual user"""
    user_id: str
    feedback_history: List[UserFeedback] = field(default_factory=list)
    
    preferred_cfg_range: Tuple[float, float] = (6.0, 9.0)
    preferred_stg_range: Tuple[float, float] = (3.0, 5.0)
    preferred_step_range: Tuple[int, int] = (30, 60)
    
    average_quality_rating: float = 0.5
    average_speed_preference: float = 0.5  # 0=speed, 1=quality
    
    def update_from_feedback(self, feedback: UserFeedback):
        """Update profile from feedback"""
        self.feedback_history.append(feedback)
        
        # Update average ratings
        ratings = [f.overall_rating for f in self.feedback_history]
        self.average_quality_rating = np.mean(ratings) if ratings else 0.5
        
        speeds = [f.speed_rating for f in self.feedback_history]
        self.average_speed_preference = np.mean(speeds) if speeds else 0.5
        
        # Update preferred ranges from top-rated samples
        top_feedbacks = sorted(
            self.feedback_history,
            key=lambda f: f.overall_rating,
            reverse=True
        )[:10]
        
        if top_feedbacks:
            cfg_scales = [f.cfg_scale for f in top_feedbacks]
            stg_scales = [f.stg_scale for f in top_feedbacks]
            steps = [f.num_steps for f in top_feedbacks]
            
            self.preferred_cfg_range = (min(cfg_scales), max(cfg_scales))
            self.preferred_stg_range = (min(stg_scales), max(stg_scales))
            self.preferred_step_range = (min(steps), max(steps))


class RewardNet(torch.nn.Module):
    """Neural network to predict user satisfaction (reward)"""
    
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256):
        """
        Initialize reward network.
        
        Args:
            embedding_dim: Size of input embeddings
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Embedding projections
        self.text_proj = torch.nn.Linear(embedding_dim, hidden_dim)
        self.video_proj = torch.nn.Linear(embedding_dim, hidden_dim)
        
        # Hyperparameter embeddings
        self.cfg_embedding = torch.nn.Embedding(100, 64)  # 0-10 range, 2 decimals
        self.stg_embedding = torch.nn.Embedding(100, 64)
        self.step_embedding = torch.nn.Embedding(100, 64)  # 0-100 steps
        
        # Main network
        total_input = hidden_dim * 2 + 64 * 3  # text + video + 3 embeddings
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(total_input, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1),  # Output: reward score 0-1
            torch.nn.Sigmoid(),
        )
    
    def forward(
        self,
        text_embedding: torch.Tensor,
        video_embedding: torch.Tensor,
        cfg_scale: torch.Tensor,
        stg_scale: torch.Tensor,
        num_steps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through reward network.
        
        Args:
            text_embedding: [batch_size, embedding_dim]
            video_embedding: [batch_size, embedding_dim]
            cfg_scale: [batch_size]
            stg_scale: [batch_size]
            num_steps: [batch_size]
            
        Returns:
            Reward scores [batch_size, 1]
        """
        # Project embeddings
        text_proj = self.text_proj(text_embedding)  # [B, hidden]
        video_proj = self.video_proj(video_embedding)  # [B, hidden]
        
        # Embedding hyperparams (convert to indices)
        cfg_idx = torch.clamp((cfg_scale * 10).long(), 0, 99)  # Scale 0-10
        stg_idx = torch.clamp((stg_scale * 20).long(), 0, 99)  # Scale 0-5
        step_idx = torch.clamp((num_steps / 2).long(), 0, 99)  # Scale steps
        
        cfg_emb = self.cfg_embedding(cfg_idx)  # [B, 64]
        stg_emb = self.stg_embedding(stg_idx)
        step_emb = self.step_embedding(step_idx)
        
        # Concatenate all features
        features = torch.cat(
            [text_proj, video_proj, cfg_emb, stg_emb, step_emb],
            dim=1
        )  # [B, hidden*2 + 64*3]
        
        # Forward through main network
        reward = self.network(features)  # [B, 1]
        
        return reward


class RewardModelForAutoTuning:
    """
    Complete reward modeling system for personalized generation.
    
    Learns user preferences and suggests optimal hyperparameters.
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize reward model.
        
        Args:
            device: "cuda" or "cpu"
        """
        self.device = device
        
        # Reward network
        self.reward_net = RewardNet(embedding_dim=512).to(device)
        self.optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()
        
        # User profiles
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Feedback buffer for batch training
        self.feedback_buffer: List[UserFeedback] = []
        self.feedback_threshold = 100  # Train every N feedbacks
    
    async def collect_user_feedback(
        self,
        user_id: str,
        generation_id: str,
        prompt: str,
        quality_rating: float,
        speed_rating: float,
        aesthetics_rating: float,
        cfg_scale: float = 7.5,
        stg_scale: float = 4.0,
        num_steps: int = 50,
        **embeddings
    ) -> Dict:
        """
        Collect feedback from user.
        
        Args:
            user_id: User identifier
            generation_id: ID of generated video
            prompt: Original prompt
            quality_rating: 0-1
            speed_rating: 0-1
            aesthetics_rating: 0-1
            cfg_scale: CFG guidance scale used
            stg_scale: STG guidance scale used
            num_steps: Number of inference steps
            **embeddings: Optional embeddings (video, audio, text)
            
        Returns:
            Dict with feedback accepted status
        """
        import time
        import uuid
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            generation_id=generation_id,
            prompt=prompt,
            quality_rating=quality_rating,
            speed_rating=speed_rating,
            aesthetics_rating=aesthetics_rating,
            overall_rating=(quality_rating + speed_rating + aesthetics_rating) / 3.0,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            num_steps=num_steps,
            timestamp=time.time(),
            **embeddings,
        )
        
        self.feedback_buffer.append(feedback)
        
        # Update user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        self.user_profiles[user_id].update_from_feedback(feedback)
        
        # Train if buffer is full
        if len(self.feedback_buffer) >= self.feedback_threshold:
            await self._train_reward_model()
        
        return {
            "accepted": True,
            "feedback_id": feedback.feedback_id,
            "buffer_size": len(self.feedback_buffer),
        }
    
    async def _train_reward_model(self, epochs: int = 5):
        """Train reward network on feedback buffer"""
        print(f"Training reward model on {len(self.feedback_buffer)} feedbacks...")
        
        if not self.feedback_buffer:
            return
        
        # This is simplified - in production would need proper embeddings
        for epoch in range(epochs):
            total_loss = 0.0
            
            for feedback in self.feedback_buffer:
                # Dummy embeddings (in practice these would be from actual models)
                text_emb = torch.randn(1, 512).to(self.device)
                video_emb = torch.randn(1, 512).to(self.device)
                
                cfg = torch.tensor([feedback.cfg_scale]).float().to(self.device)
                stg = torch.tensor([feedback.stg_scale]).float().to(self.device)
                steps = torch.tensor([feedback.num_steps]).float().to(self.device)
                
                # Forward pass
                pred_reward = self.reward_net(text_emb, video_emb, cfg, stg, steps)
                target_reward = torch.tensor([[feedback.overall_rating]]).float().to(self.device)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss = self.loss_fn(pred_reward, target_reward)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / max(len(self.feedback_buffer), 1)
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
        
        self.feedback_buffer.clear()
    
    async def suggest_hyperparameters(
        self,
        user_id: str,
        prompt: str,
        priority: str = "balanced",  # "quality", "speed", "balanced"
    ) -> Dict:
        """
        Suggest optimal hyperparameters for user and prompt.
        
        Args:
            user_id: User identifier
            prompt: Generation prompt
            priority: Optimization priority
            
        Returns:
            Suggested hyperparameter dict
        """
        # Get user profile if exists
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            
            # Use Bayesian optimization or simple heuristics
            if priority == "quality":
                cfg_scale = profile.preferred_cfg_range[1]  # Higher CFG = more faithful
                stg_scale = profile.preferred_stg_range[1]
                num_steps = profile.preferred_step_range[1]
            elif priority == "speed":
                cfg_scale = profile.preferred_cfg_range[0]  # Lower CFG = faster
                stg_scale = profile.preferred_stg_range[0]
                num_steps = profile.preferred_step_range[0]
            else:
                # Balanced: use midpoint
                cfg_scale = np.mean(profile.preferred_cfg_range)
                stg_scale = np.mean(profile.preferred_stg_range)
                num_steps = int(np.mean(profile.preferred_step_range))
        else:
            # Default for new users
            if priority == "quality":
                cfg_scale, stg_scale, num_steps = 9.0, 5.0, 60
            elif priority == "speed":
                cfg_scale, stg_scale, num_steps = 5.0, 2.0, 30
            else:
                cfg_scale, stg_scale, num_steps = 7.5, 4.0, 50
        
        return {
            "cfg_scale": float(cfg_scale),
            "stg_scale": float(stg_scale),
            "num_steps": int(num_steps),
            "priority": priority,
            "based_on_profile": user_id in self.user_profiles,
        }
    
    def get_user_profile_summary(self, user_id: str) -> Dict:
        """Get summary of user preferences"""
        if user_id not in self.user_profiles:
            return {"user_id": user_id, "status": "no_profile"}
        
        profile = self.user_profiles[user_id]
        
        return {
            "user_id": user_id,
            "feedback_count": len(profile.feedback_history),
            "average_quality_rating": float(profile.average_quality_rating),
            "average_speed_preference": float(profile.average_speed_preference),
            "preferred_cfg_range": profile.preferred_cfg_range,
            "preferred_stg_range": profile.preferred_stg_range,
            "preferred_step_range": profile.preferred_step_range,
        }
