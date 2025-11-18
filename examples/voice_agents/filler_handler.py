"""
LiveKit Voice Interruption Handler
Intelligent filler word detection to prevent false interruptions
"""

import asyncio
import logging
from typing import List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


class InterruptionType(Enum):
    """Classification of speech interruptions"""
    FILLER_ONLY = "filler_only"
    REAL_INTERRUPTION = "real_interruption"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class InterruptionEvent:
    """Represents a detected interruption event"""
    transcript: str
    classification: InterruptionType
    confidence: float
    agent_was_speaking: bool
    timestamp: float
    should_interrupt: bool


class FillerWordHandler:
    """
    Handles intelligent filtering of filler words during agent speech.
    """
    
    def __init__(
        self,
        ignored_words: Optional[List[str]] = None,
        confidence_threshold: float = 0.6,
        enable_logging: bool = True
    ):
        """
        Initialize the filler word handler.
        """
        self.ignored_words: Set[str] = set(ignored_words or [
            'uh', 'um', 'umm', 'hmm', 'hm', 'ah', 'haan', 
            'mhmm', 'uh-huh', 'mm-hmm', 'er', 'erm'
        ])
        
        self.confidence_threshold = confidence_threshold
        self.enable_logging = enable_logging
        self._agent_speaking = False
        self._lock: Optional[asyncio.Lock] = None
        
        # Statistics for debugging
        self.stats = {
            'total_events': 0,
            'fillers_ignored': 0,
            'real_interruptions': 0,
            'low_confidence_ignored': 0
        }
        
        logger.info(f"FillerWordHandler initialized with {len(self.ignored_words)} ignored words")
    
    def set_agent_speaking(self, speaking: bool):
        """
        Update the agent's speaking state.
        """
        self._agent_speaking = speaking
        if self.enable_logging:
            logger.debug(f"Agent speaking state: {speaking}")
    
    def is_agent_speaking(self) -> bool:
        """Check if agent is currently speaking"""
        return self._agent_speaking
    
    def update_ignored_words(self, words: List[str]):
        """
        Dynamically update the list of ignored filler words.
        """
        self.ignored_words = set(w.lower().strip() for w in words)
        logger.info(f"Updated ignored words: {self.ignored_words}")
    
    def add_ignored_words(self, words: List[str]):
        """Add words to the ignored list without replacing existing ones"""
        self.ignored_words.update(w.lower().strip() for w in words)
        logger.info(f"Added words. Total ignored: {len(self.ignored_words)}")
    
    def remove_ignored_words(self, words: List[str]):
        """Remove words from the ignored list"""
        self.ignored_words.difference_update(w.lower().strip() for w in words)
        logger.info(f"Removed words. Total ignored: {len(self.ignored_words)}")
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s-]', '', text)
        return text
    
    def _classify_interruption(self, transcript: str) -> InterruptionType:
        """
        Classify whether the transcript contains only fillers or real content.
        """
        normalized = self._normalize_text(transcript)
        words = normalized.split()
        
        if not words:
            return InterruptionType.UNKNOWN
        
        # Check how many words are fillers
        filler_count = sum(1 for word in words if word in self.ignored_words)
        
        if filler_count == len(words):
            # All words are fillers
            return InterruptionType.FILLER_ONLY
        elif filler_count > 0:
            # Mix of fillers and real words
            return InterruptionType.MIXED
        else:
            # No fillers, all real content
            return InterruptionType.REAL_INTERRUPTION
    
    async def process_interruption(
        self,
        transcript: str,
        confidence: float = 1.0,
        timestamp: Optional[float] = None
    ) -> InterruptionEvent:
        """
        Process a potential interruption event.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        
        async with self._lock:
            self.stats['total_events'] += 1
            
            if timestamp is None:
                timestamp = asyncio.get_event_loop().time()
            
            # Low confidence speech is ignored during agent speaking
            if confidence < self.confidence_threshold and self._agent_speaking:
                self.stats['low_confidence_ignored'] += 1
                if self.enable_logging:
                    logger.debug(
                        f"Ignored low confidence ({confidence:.2f}): '{transcript}'"
                    )
                
                return InterruptionEvent(
                    transcript=transcript,
                    classification=InterruptionType.UNKNOWN,
                    confidence=confidence,
                    agent_was_speaking=self._agent_speaking,
                    timestamp=timestamp,
                    should_interrupt=False
                )
            
            # Classify the interruption
            classification = self._classify_interruption(transcript)
            
            # Decision logic
            should_interrupt = True
            
            if self._agent_speaking:
                if classification == InterruptionType.FILLER_ONLY:
                    # Ignore filler-only interruptions during agent speech
                    should_interrupt = False
                    self.stats['fillers_ignored'] += 1
                    
                    if self.enable_logging:
                        logger.info(
                            f"IGNORED FILLER: '{transcript}' "
                            f"(confidence: {confidence:.2f})"
                        )
                else:
                    # Real content or mixed - interrupt the agent
                    should_interrupt = True
                    self.stats['real_interruptions'] += 1
                    
                    if self.enable_logging:
                        logger.info(
                            f"REAL INTERRUPTION: '{transcript}' "
                            f"(type: {classification.value}, confidence: {confidence:.2f})"
                        )
            else:
                # Agent not speaking - accept all input including fillers
                if self.enable_logging:
                    logger.debug(
                        f"Accepted (agent quiet): '{transcript}' "
                        f"(type: {classification.value})"
                    )
            
            return InterruptionEvent(
                transcript=transcript,
                classification=classification,
                confidence=confidence,
                agent_was_speaking=self._agent_speaking,
                timestamp=timestamp,
                should_interrupt=should_interrupt
            )
    
    def get_stats(self) -> dict:
        """Get statistics about processed interruptions"""
        return {
            **self.stats,
            'ignore_rate': (
                self.stats['fillers_ignored'] / self.stats['total_events']
                if self.stats['total_events'] > 0 else 0
            )
        }
    
    def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            'total_events': 0,
            'fillers_ignored': 0,
            'real_interruptions': 0,
            'low_confidence_ignored': 0
        }
        logger.info("Statistics reset")


# Example integration with LiveKit Agent
class LiveKitAgentWithFillerHandling:
    
    def __init__(self, ignored_words: Optional[List[str]] = None):
        self.filler_handler = FillerWordHandler(ignored_words=ignored_words)
        self.tts_active = False
    
    async def on_tts_started(self):
        """Called when agent starts speaking"""
        self.tts_active = True
        self.filler_handler.set_agent_speaking(True)
        logger.info("Agent started speaking")
    
    async def on_tts_stopped(self):
        """Called when agent stops speaking"""
        self.tts_active = False
        self.filler_handler.set_agent_speaking(False)
        logger.info("Agent stopped speaking")
    
    async def on_user_transcription(
        self,
        transcript: str,
        confidence: float = 1.0
    ):
        """
        Called when user speech is transcribed.
        """
        # Process through filler handler
        event = await self.filler_handler.process_interruption(
            transcript=transcript,
            confidence=confidence
        )
        
        if event.should_interrupt and self.tts_active:
            # Stop the agent's TTS
            await self.stop_tts()
            logger.info("Agent interrupted by user")
        
        return event
    
    async def stop_tts(self):
        """Stop agent's text-to-speech output"""
        # This would call your actual LiveKit TTS cancellation
        self.tts_active = False
        self.filler_handler.set_agent_speaking(False)
        logger.info("Stopping TTS...")


# Example usage and testing
async def demo_scenarios():
    """Demonstrate various interruption scenarios"""
    
    print("\n" + "="*60)
    print("LiveKit Filler Word Handler - Demo Scenarios")
    print("="*60 + "\n")
    
    agent = LiveKitAgentWithFillerHandling()
    
    # Scenario 1: Filler during agent speech
    print("Scenario 1: Filler while agent speaks")
    await agent.on_tts_started()
    await asyncio.sleep(0.1)
    event = await agent.on_user_transcription("umm", confidence=0.95)
    print(f"   Should interrupt: {event.should_interrupt} ✓\n")
    await agent.on_tts_stopped()
    
    # Scenario 2: Real interruption
    print("Scenario 2: Real interruption")
    await agent.on_tts_started()
    await asyncio.sleep(0.1)
    event = await agent.on_user_transcription("wait one second", confidence=0.92)
    print(f"   Should interrupt: {event.should_interrupt}\n")
    
    # Scenario 3: Filler while agent quiet
    print("Scenario 3: Filler while agent quiet")
    event = await agent.on_user_transcription("umm", confidence=0.88)
    print(f"   Should interrupt: {event.should_interrupt}\n")
    
    # Scenario 4: Mixed filler and command
    print("Scenario 4: Mixed filler + command")
    await agent.on_tts_started()
    await asyncio.sleep(0.1)
    event = await agent.on_user_transcription("umm okay stop", confidence=0.90)
    print(f"Should interrupt: {event.should_interrupt}\n")
    
    # Scenario 5: Low confidence murmur
    print("Scenario 5: Low confidence background noise")
    event = await agent.on_user_transcription("hmm yeah", confidence=0.45)
    print(f"   Should interrupt: {event.should_interrupt}\n")
    
    # Print statistics
    print("="*60)
    print("Statistics:")
    print("="*60)
    stats = agent.filler_handler.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_scenarios())
