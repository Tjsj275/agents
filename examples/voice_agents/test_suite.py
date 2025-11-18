"""
Test Suite for LiveKit Filler Word Handler
Comprehensive testing for all scenarios and edge cases
"""

import asyncio
import unittest
from typing import List, Tuple
import sys
import time
import nest_asyncio

nest_asyncio.apply()

# Import the handler (adjust path as needed)
from filler_handler import (
    FillerWordHandler, 
    InterruptionType, 
    InterruptionEvent
)


class TestFillerWordHandler(unittest.IsolatedAsyncioTestCase):
    """Unit tests for FillerWordHandler"""
    
    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.handler = FillerWordHandler(
            ignored_words=['uh', 'um', 'umm', 'hmm', 'haan'],
            confidence_threshold=0.6,
            enable_logging=False
        )
    
    # ========== Classification Tests ==========
    
    async def test_filler_only_classification(self):
        """Test that filler-only input is correctly classified"""
        event = await self.handler.process_interruption("umm")
        self.assertEqual(event.classification, InterruptionType.FILLER_ONLY)
        
        event = await self.handler.process_interruption("uh hmm")
        self.assertEqual(event.classification, InterruptionType.FILLER_ONLY)
    
    async def test_real_interruption_classification(self):
        """Test that real speech is correctly classified"""
        event = await self.handler.process_interruption("wait one second")
        self.assertEqual(event.classification, InterruptionType.REAL_INTERRUPTION)
        
        event = await self.handler.process_interruption("stop")
        self.assertEqual(event.classification, InterruptionType.REAL_INTERRUPTION)
    
    async def test_mixed_classification(self):
        """Test that mixed filler + real speech is correctly classified"""
        event = await self.handler.process_interruption("umm okay stop")
        self.assertEqual(event.classification, InterruptionType.MIXED)
        
        event = await self.handler.process_interruption("uh wait a second")
        self.assertEqual(event.classification, InterruptionType.MIXED)
    
    # ========== Agent Speaking State Tests ==========
    
    async def test_filler_ignored_when_agent_speaking(self):
        """Filler should be ignored when agent is speaking"""
        self.handler.set_agent_speaking(True)
        event = await self.handler.process_interruption("umm", confidence=0.9)
        
        self.assertFalse(event.should_interrupt)
        self.assertTrue(event.agent_was_speaking)
    
    async def test_real_interrupt_when_agent_speaking(self):
        """Real interruption should stop agent"""
        self.handler.set_agent_speaking(True)
        event = await self.handler.process_interruption("wait stop", confidence=0.9)
        
        self.assertTrue(event.should_interrupt)
        self.assertTrue(event.agent_was_speaking)
    
    async def test_filler_accepted_when_agent_quiet(self):
        """Filler should be accepted when agent is quiet"""
        self.handler.set_agent_speaking(False)
        event = await self.handler.process_interruption("umm", confidence=0.9)
        
        # When agent is quiet, all speech is accepted
        self.assertTrue(event.should_interrupt)
        self.assertFalse(event.agent_was_speaking)
    
    async def test_mixed_content_interrupts_agent(self):
        """Mixed filler + command should interrupt"""
        self.handler.set_agent_speaking(True)
        event = await self.handler.process_interruption("umm no wait", confidence=0.9)
        
        self.assertTrue(event.should_interrupt)
        self.assertEqual(event.classification, InterruptionType.MIXED)
    
    # ========== Confidence Threshold Tests ==========
    
    async def test_low_confidence_ignored_during_speech(self):
        """Low confidence should be ignored during agent speech"""
        self.handler.set_agent_speaking(True)
        event = await self.handler.process_interruption("hmm yeah", confidence=0.3)
        
        self.assertFalse(event.should_interrupt)
    
    async def test_high_confidence_processed(self):
        """High confidence should be processed"""
        self.handler.set_agent_speaking(True)
        event = await self.handler.process_interruption("stop", confidence=0.95)
        
        self.assertTrue(event.should_interrupt)
    
    async def test_confidence_threshold_boundary(self):
        """Test confidence at exact threshold"""
        self.handler.set_agent_speaking(True)
        
        # Just below threshold
        event = await self.handler.process_interruption("stop", confidence=0.59)
        self.assertFalse(event.should_interrupt)
        
        # At threshold
        event = await self.handler.process_interruption("stop", confidence=0.60)
        self.assertTrue(event.should_interrupt)
    
    # ========== Dynamic Configuration Tests ==========
    
    async def test_update_ignored_words(self):
        """Test updating the ignored words list"""
        # Initially "okay" is not ignored
        event = await self.handler.process_interruption("okay")
        self.assertNotEqual(event.classification, InterruptionType.FILLER_ONLY)
        
        # Add "okay" to ignored list
        self.handler.add_ignored_words(['okay'])
        event = await self.handler.process_interruption("okay")
        self.assertEqual(event.classification, InterruptionType.FILLER_ONLY)
    
    async def test_remove_ignored_words(self):
        """Test removing words from ignored list"""
        # Initially "umm" is ignored
        event = await self.handler.process_interruption("umm")
        self.assertEqual(event.classification, InterruptionType.FILLER_ONLY)
        
        # Remove "umm" from ignored list
        self.handler.remove_ignored_words(['umm'])
        event = await self.handler.process_interruption("umm")
        self.assertNotEqual(event.classification, InterruptionType.FILLER_ONLY)
    
    # ========== Text Normalization Tests ==========
    
    async def test_case_insensitivity(self):
        """Test that classification is case-insensitive"""
        tests = ["UMM", "Umm", "uMm", "umm"]
        for text in tests:
            event = await self.handler.process_interruption(text)
            self.assertEqual(event.classification, InterruptionType.FILLER_ONLY)
    
    async def test_punctuation_handling(self):
        """Test that punctuation is handled correctly"""
        event = await self.handler.process_interruption("umm, uh...")
        self.assertEqual(event.classification, InterruptionType.FILLER_ONLY)
        
        event = await self.handler.process_interruption("wait!")
        self.assertEqual(event.classification, InterruptionType.REAL_INTERRUPTION)
    
    async def test_whitespace_handling(self):
        """Test handling of extra whitespace"""
        event = await self.handler.process_interruption("  umm   hmm  ")
        self.assertEqual(event.classification, InterruptionType.FILLER_ONLY)
    
    # ========== Statistics Tests ==========
    
    async def test_statistics_tracking(self):
        """Test that statistics are tracked correctly"""
        self.handler.reset_stats()
        self.handler.set_agent_speaking(True)
        
        # Process some events
        await self.handler.process_interruption("umm", confidence=0.9)
        await self.handler.process_interruption("stop", confidence=0.9)
        await self.handler.process_interruption("uh", confidence=0.3)
        
        stats = self.handler.get_stats()
        
        self.assertEqual(stats['total_events'], 3)
        self.assertEqual(stats['fillers_ignored'], 1)
        self.assertEqual(stats['real_interruptions'], 1)
        self.assertEqual(stats['low_confidence_ignored'], 1)
    
    # ========== Edge Cases ==========
    
    async def test_empty_string(self):
        """Test handling of empty input"""
        event = await self.handler.process_interruption("")
        self.assertEqual(event.classification, InterruptionType.UNKNOWN)
    
    async def test_only_punctuation(self):
        """Test input with only punctuation"""
        event = await self.handler.process_interruption("...")
        self.assertEqual(event.classification, InterruptionType.UNKNOWN)
    
    async def test_multilingual_fillers(self):
        """Test with multiple languages"""
        # Add Hindi filler
        self.handler.add_ignored_words(['हाँ', 'acha'])
        
        event = await self.handler.process_interruption("acha")
        self.assertEqual(event.classification, InterruptionType.FILLER_ONLY)


class IntegrationTestScenarios:
    
    def __init__(self):
        self.handler = FillerWordHandler(enable_logging=True)
        self.results: List[Tuple[str, bool]] = []
    
    async def run_scenario(self, name: str, description: str, test_func):
        """Run a single scenario and track results"""
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{description}")
        print(f"{'='*60}")
        
        try:
            success = await test_func()
            self.results.append((name, success))
            
            if success:
                print(f"PASSED\n")
            else:
                print(f"FAILED\n")
                
        except Exception as e:
            print(f"ERROR: {e}\n")
            self.results.append((name, False))
    
    async def scenario_1_filler_during_speech(self):
        """User says filler while agent is speaking"""
        self.handler.set_agent_speaking(True)
        await asyncio.sleep(0.1)  # Simulate agent speaking
        
        event = await self.handler.process_interruption("umm", confidence=0.95)
        
        # Should NOT interrupt
        return not event.should_interrupt
    
    async def scenario_2_real_interruption(self):
        """User interrupts with real speech"""
        self.handler.set_agent_speaking(True)
        await asyncio.sleep(0.1)
        
        event = await self.handler.process_interruption("wait one second", confidence=0.92)
        
        # Should interrupt
        return event.should_interrupt
    
    async def scenario_3_filler_when_quiet(self):
        """User says filler when agent is quiet"""
        self.handler.set_agent_speaking(False)
        
        event = await self.handler.process_interruption("umm", confidence=0.88)
        
        # Should be processed (not interrupt agent, but accepted)
        return event.should_interrupt  # True because agent not speaking
    
    async def scenario_4_mixed_content(self):
        """User says filler + command"""
        self.handler.set_agent_speaking(True)
        await asyncio.sleep(0.1)
        
        event = await self.handler.process_interruption("umm okay stop", confidence=0.90)
        
        # Should interrupt (contains real command)
        return event.should_interrupt
    
    async def scenario_5_low_confidence_murmur(self):
        """Background noise with low confidence"""
        self.handler.set_agent_speaking(True)
        await asyncio.sleep(0.1)
        
        event = await self.handler.process_interruption("hmm yeah", confidence=0.45)
        
        # Should be ignored
        return not event.should_interrupt
    
    async def scenario_6_rapid_turn_taking(self):
        """Rapid back-and-forth conversation"""
        results = []
        
        # Agent speaks
        self.handler.set_agent_speaking(True)
        e1 = await self.handler.process_interruption("umm", confidence=0.9)
        results.append(not e1.should_interrupt)  # Should ignore
        
        # User interrupts
        e2 = await self.handler.process_interruption("hold on", confidence=0.9)
        results.append(e2.should_interrupt)  # Should interrupt
        
        # Agent stops
        self.handler.set_agent_speaking(False)
        
        # User continues
        e3 = await self.handler.process_interruption("umm let me think", confidence=0.9)
        results.append(e3.should_interrupt)  # Should be accepted
        
        return all(results)
    
    async def scenario_7_multiple_fillers(self):
        """Multiple fillers in sequence"""
        self.handler.set_agent_speaking(True)
        
        event = await self.handler.process_interruption("uh umm hmm", confidence=0.85)
        
        # All fillers - should ignore
        return not event.should_interrupt
    
    async def run_all(self):
        """Run all integration scenarios"""
        print("\n" + "="*60)
        print("LIVEKIT FILLER HANDLER - INTEGRATION TESTS")
        print("="*60)
        
        await self.run_scenario(
            "Scenario 1",
            "Filler while agent speaks → Should IGNORE",
            self.scenario_1_filler_during_speech
        )
        
        await self.run_scenario(
            "Scenario 2",
            "Real interruption → Should STOP agent",
            self.scenario_2_real_interruption
        )
        
        await self.run_scenario(
            "Scenario 3",
            "Filler while agent quiet → Should REGISTER",
            self.scenario_3_filler_when_quiet
        )
        
        await self.run_scenario(
            "Scenario 4",
            "Mixed filler + command → Should STOP agent",
            self.scenario_4_mixed_content
        )
        
        await self.run_scenario(
            "Scenario 5",
            "Low confidence murmur → Should IGNORE",
            self.scenario_5_low_confidence_murmur
        )
        
        await self.run_scenario(
            "Scenario 6",
            "Rapid turn-taking → Should handle correctly",
            self.scenario_6_rapid_turn_taking
        )
        
        await self.run_scenario(
            "Scenario 7",
            "Multiple fillers → Should IGNORE",
            self.scenario_7_multiple_fillers
        )
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for _, success in self.results if success)
        total = len(self.results)
        
        for name, success in self.results:
            status = "PASS" if success else "FAIL"
            print(f"{status} - {name}")
        
        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
        print(f"{'='*60}\n")
        
        # Print handler statistics
        stats = self.handler.get_stats()
        print("Handler Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2%}" if 'rate' in key else f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        return passed == total


def run_unit_tests():
    """Run unit tests"""
    print("\nRunning Unit Tests...\n")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFillerWordHandler)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


async def run_integration_tests():
    """Run integration tests"""
    scenarios = IntegrationTestScenarios()
    return await scenarios.run_all()


async def main():
    """Main test runner"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--integration':
        # Run integration tests only
        success = await run_integration_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == '--unit':
        # Run unit tests only
        success = run_unit_tests()
    else:
        # Run both
        print("\n" + "="*60)
        print("RUNNING ALL TESTS")
        print("="*60)
        
        unit_success = run_unit_tests()
        integration_success = await run_integration_tests()
        
        success = unit_success and integration_success
        
        print("\n" + "="*60)
        print("🎯 FINAL RESULTS")
        print("="*60)
        print(f"Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
        print(f"Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")
        print(f"Overall: {'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
        print("="*60 + "\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == '--unit':
        sys.argv.pop(1)
        success = run_unit_tests()
        sys.exit(0 if success else 1)

    if len(sys.argv) > 1 and sys.argv[1] == '--integration':
        sys.argv.pop(1)
        import asyncio as _asyncio
        success = _asyncio.run(run_integration_tests())
        sys.exit(0 if success else 1)

    unit_success = run_unit_tests()
    integration_success = asyncio.run(run_integration_tests())
    overall = unit_success and integration_success

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Unit Tests: {'PASSED' if unit_success else 'FAILED'}")
    print(f"Integration Tests: {'PASSED' if integration_success else 'FAILED'}")
    print(f"Overall: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
    print("="*60 + "\n")

    sys.exit(0 if overall else 1)


