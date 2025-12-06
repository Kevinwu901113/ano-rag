
from utils.answer_cleaner import clean_model_answer, _strip_reasoning

def sanity_check():
    print("Running sanity checks for answer_cleaner...")
    
    # Case 1: Standard removal
    s1 = "<think>blah blah</think> Final answer."
    res1 = _strip_reasoning(s1)
    print(f"Case 1 Input: '{s1}'")
    print(f"Case 1 Output: '{res1}'")
    assert res1 == "Final answer."
    
    # Case 2: Only open tag (should truncate rest)
    s2 = "<think>Thinking indefinitely... Answer is 42"
    res2 = _strip_reasoning(s2)
    print(f"Case 2 Input: '{s2}'")
    print(f"Case 2 Output: '{res2}'")
    assert res2 == ""
    
    # Case 3: Multiple blocks
    s3 = "Start <think>thought 1</think> Middle <think>thought 2</think> End"
    res3 = _strip_reasoning(s3)
    print(f"Case 3 Input: '{s3}'")
    print(f"Case 3 Output: '{res3}'")
    assert res3 == "Start  Middle  End" # Note spaces might be preserved
    
    # Case 4: No tags
    s4 = "Just an answer."
    res4 = _strip_reasoning(s4)
    print(f"Case 4 Input: '{s4}'")
    print(f"Case 4 Output: '{res4}'")
    assert res4 == "Just an answer."
    
    # Case 5: Empty
    s5 = ""
    res5 = _strip_reasoning(s5)
    print(f"Case 5 Input: '{s5}'")
    print(f"Case 5 Output: '{res5}'")
    assert res5 == ""

    print("All sanity checks passed!")

if __name__ == "__main__":
    sanity_check()
