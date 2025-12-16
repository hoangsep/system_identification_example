import numpy as np
import sys

# Mock class to simulate NeuralMPC behavior relevant to path wrapping
class MockMPC:
    def __init__(self, path_s):
        self.path_s = np.array(path_s)
        self.total_path_length = self.path_s[-1] if len(self.path_s) > 0 else 0.0

    def index_ahead_by_distance(self, start_idx, dist):
        if len(self.path_s) == 0: return 0
        
        # Proposed new logic
        current_s = self.path_s[start_idx]
        target_s = (current_s + dist) % self.total_path_length
        
        # If target_s wrapped around and is smaller than current_s (and we were near the end),
        # searchsorted works normally because path_s is sorted.
        # But wait, searchsorted finds the index where element should be inserted to maintain order.
        # If target_s is small (wrapped), it will find an index near the beginning.
        
        idx = np.searchsorted(self.path_s, target_s)
        return min(idx, len(self.path_s)-1)

def test_wrapping():
    # Setup a simple path: distances 0, 10, 20, 30, 40
    path_s = [0.0, 10.0, 20.0, 30.0, 40.0]
    mpc = MockMPC(path_s)
    
    print(f"Path S: {path_s}")
    print(f"Total Length: {mpc.total_path_length}")
    
    # Test 1: Normal lookahead (Start 0, dist 15 -> Target 15 -> Index 2 (20.0))
    idx = mpc.index_ahead_by_distance(0, 15.0)
    print(f"Test 1 (Normal): Start 0, Dist 15 => Index {idx} (val {path_s[idx]})")
    assert idx == 2, f"Expected index 2, got {idx}"
    
    # Test 2: Wrap around (Start 3 (30.0), dist 15 -> Target 45 % 40 = 5.0 -> Index 1 (10.0))
    # Wait, 30+15 = 45. 45 % 40 = 5.
    # searchsorted([0, 10...], 5) gives index 1 (since 5 is between 0 and 10, insert at 1)
    # This seems correct for finding the *next* point.
    idx = mpc.index_ahead_by_distance(3, 15.0)
    print(f"Test 2 (Wrap): Start 3 (30.0), Dist 15 => Target 5.0 => Index {idx} (val {path_s[idx]})")
    assert idx == 1, f"Expected index 1, got {idx}"
    
    # Test 3: Exact Wrap (Start 4 (40.0), dist 10 -> Target 50 % 40 = 10.0 -> Index 1 (10.0))
    # searchsorted for 10.0 in [0, 10, ...] puts it at index 1 (left side by default)
    idx = mpc.index_ahead_by_distance(4, 10.0)
    print(f"Test 3 (Exact): Start 4 (40.0), Dist 10 => Target 10.0 => Index {idx} (val {path_s[idx]})")
    assert idx == 1, f"Expected index 1, got {idx}"

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_wrapping()
