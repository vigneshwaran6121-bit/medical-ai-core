from typing import List


def contains_duplicate(nums: list[int]):
    unique_elements = set(nums)

    if len(unique_elements) < len(nums):
        return True
    else:
        return False


if __name__ == "__main__":
    test_1 = [1, 2, 3, 4]
    test_2 = [1, 2, 3, 1]

    print(f"test 1 ([1,2,3,4]): {contains_duplicate(test_1)}")
    print(f"test 2 ([1,2,3,1]): {contains_duplicate(test_2)}")
