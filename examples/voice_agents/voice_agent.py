"""
Complete LiveKit Voice Agent with Filler Word Handling
A practical, working example of integrating the filler handler
"""

import asyncio
import logging
import os
from typing import Optional

LIVEKIT_AVAILABLE = True
try:
    from livekit import agents, rtc
    from livekit.agents import JobContext, WorkerOptions, cli
    from livekit.plugins import openai, silero
except Exception:
    LIVEKIT_AVAILABLE = False
    agents = None
    rtc = None
    JobContext = object
    WorkerOptions = type("WorkerOptions", (), {})
    cli = type("CLI", (), {"run_app": staticmethod(lambda opts: None)})

    # Minimal stub replacements for openai TTS/STT used in example/test mode
    class _DummyAudioChunk:
        def __init__(self):
            self.data = b""

    class _DummyTTS:
        async def synthesize(self, text):
            # yield a single dummy chunk
            yield _DummyAudioChunk()

    class _DummySTTEvent:
        def __init__(self, transcript="", confidence=1.0, is_final=True):
            self.is_final = is_final
            self.alternatives = [type("Alt", (), {"transcript": transcript, "confidence": confidence})]

    class _DummySTT:
        async def stream(self):
            # no events by default; callers can replace with their own test harness
            if False:
                yield _DummySTTEvent()

    openai = type("openai", (), {"TTS": lambda voice=None: _DummyTTS(), "STT": lambda: _DummySTT()})
    silero = None


# Import our filler handler
from filler_handler import FillerWordHandler

logger = logging.getLogger(__name__)


class VoiceAssistant(agents.MultimodalAgent):
    
    def __init__(self, ctx: JobContext):
        super().__init__()
        self.ctx = ctx
        
        # Initialize filler handler with configurable settings
        ignored_words = os.getenv('IGNORED_FILLER_WORDS', 
                                 'uh,um,umm,hmm,haan,mhmm,ah,er').split(',')
        confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))
        
        self.filler_handler = FillerWordHandler(
            ignored_words=[w.strip() for w in ignored_words],
            confidence_threshold=confidence_threshold,
            enable_logging=True
        )
        
        # Initialize TTS and STT
        self.tts = openai.TTS(voice="alloy")
        self.stt = openai.STT()
        
        # Agent state
        self.agent_speaking = False
        self.current_tts_task: Optional[asyncio.Task] = None
        
        logger.info(f"VoiceAssistant initialized with {len(ignored_words)} filler words")
    
    async def start(self):
        """Start the voice assistant"""
        # Connect to the room
        await self.ctx.connect()
        logger.info(f"Connected to room: {self.ctx.room.name}")
        
        # Get the participant (user)
        participant = await self.ctx.wait_for_participant()
        logger.info(f"Participant joined: {participant.identity}")
        
        # Start listening for audio
        audio_stream = rtc.AudioStream(participant.tracks[0])
        
        # Set up event handlers
        self.setup_event_handlers()
        
        # Start transcription
        asyncio.create_task(self.transcribe_audio(audio_stream))
        
        # Welcome message
        await self.speak("Hello! How can I help you today?")
    
    def setup_event_handlers(self):
        """Set up event handlers for room events"""
        room = self.ctx.room
        
        # Handle participant events
        @room.on("participant_disconnected")
        def on_disconnect(participant: rtc.Participant):
            logger.info(f"Participant disconnected: {participant.identity}")
            self.cleanup()
        
        # Handle track events
        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            logger.info(f"Track subscribed: {track.kind}")
    
    async def transcribe_audio(self, audio_stream: rtc.AudioStream):
        async for event in self.stt.stream():
            if not event.is_final:
                # Skip interim results
                continue
            
            transcript = event.alternatives[0].transcript
            confidence = event.alternatives[0].confidence
            
            logger.debug(f"Transcription: '{transcript}' (confidence: {confidence:.2f})")
            
            # Process through filler handler
            interruption = await self.filler_handler.process_interruption(
                transcript=transcript,
                confidence=confidence
            )
            
            # Decide what to do based on the result
            if interruption.should_interrupt:
                # This is a real interruption
                if self.agent_speaking:
                    logger.info(f"User interrupted: '{transcript}'")
                    await self.stop_speaking()
                
                # Process the user's message
                await self.handle_user_message(transcript)
            else:
                # This was filtered as a filler
                logger.debug(f"Filtered filler: '{transcript}'")
                # Continue without interrupting
    
    async def speak(self, text: str):
        logger.info(f"Agent speaking: '{text[:50]}...'")
        
        # Update state
        self.agent_speaking = True
        self.filler_handler.set_agent_speaking(True)
        
        try:
            # Create TTS task
            self.current_tts_task = asyncio.create_task(
                self._speak_internal(text)
            )
            
            # Wait for completion
            await self.current_tts_task
            
        except asyncio.CancelledError:
            logger.info("Speech cancelled by interruption")
        
        finally:
            # Update state
            self.agent_speaking = False
            self.filler_handler.set_agent_speaking(False)
            self.current_tts_task = None
            logger.info("Agent stopped speaking")
    
    async def _speak_internal(self, text: str):
        """Internal method to perform TTS"""
        # Generate audio
        async for audio_chunk in self.tts.synthesize(text):
            # Publish audio to the room
            await self.ctx.room.local_participant.publish_data(
                audio_chunk.data,
                reliable=True
            )
            
            # Small delay to simulate real-time speech
            await asyncio.sleep(0.01)
    
    async def stop_speaking(self):
        """
        Stop the agent's current speech.
        """
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel()
            
            try:
                await self.current_tts_task
            except asyncio.CancelledError:
                pass
        
        self.agent_speaking = False
        self.filler_handler.set_agent_speaking(False)
    
    async def handle_user_message(self, message: str):
        """
        Process a user message and generate a response.
        """
        logger.info(f"Processing user message: '{message}'")
        
        # Example responses (replace with actual LLM integration)
        if "stop" in message.lower() or "wait" in message.lower():
            await self.speak("Okay, I'll stop. What would you like to know?")
        
        elif "hello" in message.lower() or "hi" in message.lower():
            await self.speak("Hello! How can I assist you?")
        
        else:
            # Default response - replace with actual LLM
            await self.speak("I understand. Let me help you with that.")
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up agent resources...")
        
        # Print final statistics
        stats = self.filler_handler.get_stats()
        logger.info("="*50)
        logger.info("Filler Handler Statistics:")
        logger.info("="*50)
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        logger.info("="*50)

# Entry point for LiveKit worker
async def entrypoint(ctx: JobContext):
    """
    Main entry point for the LiveKit agent.
    
    This is called by LiveKit's worker system.
    """
    logger.info("Starting voice assistant...")
    
    # Create and start the assistant
    assistant = VoiceAssistant(ctx)
    await assistant.start()
    
    # Keep the agent running
    await ctx.room.disconnect()


def main():
    """
    Run the agent worker.
    
    Usage:
        python voice_agent.py dev  # Development mode
        python voice_agent.py start  # Production mode
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure worker options
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=None,  # Optional: custom request handler
        
        # Worker configuration
        worker_type=agents.WorkerType.ROOM,
        
        # Resource limits
        max_retry=3,
        ws_url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
    
    # Start the worker
    logger.info("Starting LiveKit worker...")
    cli.run_app(worker_options)


if __name__ == "__main__":
    # Example of running the simple agent for testing
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "--test":
        # Run test mode
        async def test():
            agent = SimpleVoiceAgent()
            
            print("\n" + "="*60)
            print("Testing Simple Voice Agent")
            print("="*60 + "\n")
            
            # Simulate agent speaking
            await agent.on_agent_speech_started()
            
            # Test filler during speech
            await agent.on_transcription_received("umm", 0.9)
            
            # Test real interruption
            await agent.on_transcription_received("wait stop", 0.95)
            
            # Agent stops
            await agent.on_agent_speech_ended()
            
            # Test filler when quiet
            await agent.on_transcription_received("umm", 0.9)
            
            print("\n" + "="*60)
            print("Test complete!")
            print("="*60)
        
        asyncio.run(test())
    else:
        # Run the full agent
        main()

