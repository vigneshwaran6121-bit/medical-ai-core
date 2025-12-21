def ab(a, b):  # a-num b-target
    seen = {}

    for index, num in enumerate(a):
        needed = b - num

        if needed in seen:
            return [seen[needed], index]

        seen[num] = index
    return []


# --- Test Block ---
if __name__ == "__main__":
    nums = [2, 7, 11, 15]
    target = 9
    result = ab(nums, target)

    print("Input:", nums)
    print("Target:", target)
    print("Result:", result)
