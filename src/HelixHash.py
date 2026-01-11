import streamlit as st
import time
import traceback

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HelixHash | Medical DSA", page_icon="🧬", layout="wide")

# --- CUSTOM STYLES (Simple CSS for that 'LeetCode' dark mode feel) ---
st.markdown(
    """
<style>
    .stTextArea textarea {
        background-color: #1E1E1E;
        color: #D4D4D4;
        font-family: 'Courier New', Courier, monospace;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        margin-bottom: 1rem;
    }
    .fail-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- SIDEBAR: NAVIGATION ---
st.sidebar.title("🧬 HelixHash")
st.sidebar.markdown("**Master DSA with Medical Data**")
st.sidebar.write("---")
problem_selection = st.sidebar.radio(
    "Select Problem:", ["1. The Viral Rearrangement", "2. The RNA Folding Validator"]
)

# --- PROBLEM 1 DATA ---
if problem_selection == "1. The Viral Rearrangement":
    st.title("1. The Viral Rearrangement")

    # Difficulty Badges
    c1, c2, c3 = st.columns([1, 1, 8])
    with c1:
        st.markdown("![Easy](https://img.shields.io/badge/Difficulty-Easy-green)")
    with c2:
        st.markdown("![Topic](https://img.shields.io/badge/Topic-Hash_Map-blue)")

    st.write("---")

    # Layout: Left for Description, Right for Code
    col_desc, col_code = st.columns([1, 1])

    with col_desc:
        st.subheader("🧬 The Clinical Scenario")
        st.write(
            """
        You are a Bioinformatics Engineer at a virology lab. Dr. Nila has isolated a DNA fragment from a patient (`patient_seq`). 
        
        She suspects this is a variation of a known virus reference (`reference_seq`) caused by a **rearrangement mutation** (inversion or translocation).
        
        **Your Task:**
        Write a function `check_mutation(reference_seq, patient_seq)` that returns `True` if the patient sequence is a valid rearrangement (anagram) of the reference. It must contain the **exact same nucleotide counts**.
        """
        )

        st.info(
            """
        **Example 1:**
        \nInput: ref = "ATTCG", patient = "TGCTA"
        \nOutput: True
        \n(Explanation: Both have 1 A, 2 Ts, 1 C, 1 G)
        """
        )

        st.info(
            """
        **Example 2:**
        \nInput: ref = "AAACCC", patient = "AAACCG"
        \nOutput: False
        \n(Explanation: Counts do not match.)
        """
        )

    with col_code:
        st.subheader("💻 Your Solution")
        st.markdown(
            "Write your Python code below. The function must be named `check_mutation`."
        )

        # Default starter code
        default_code = """def check_mutation(reference_seq, patient_seq):
    # Write your logic here
    # Remember: You need to compare the counts of A, T, C, G
    
    return False"""

        user_code = st.text_area("Code Editor", value=default_code, height=350)

        if st.button("Run Clinical Test Cases", type="primary"):
            with st.spinner("Running sequences through the analyzer..."):
                time.sleep(1)  # Dramatic pause for effect

                try:
                    # 1. Prepare the execution environment
                    local_scope = {}

                    # 2. Execute user code safely-ish (Caution: exec() runs arbitrary code)
                    exec(user_code, {}, local_scope)

                    # 3. Check if function exists
                    if "check_mutation" not in local_scope:
                        st.error(
                            "Error: You must define a function named 'check_mutation'"
                        )
                    else:
                        func = local_scope["check_mutation"]

                        # --- THE HIDDEN TEST CASES ---
                        test_cases = [
                            {
                                "input": ("GATTACA", "ACATTAG"),
                                "expected": True,
                                "desc": "Standard Rearrangement",
                            },
                            {
                                "input": ("ATC", "ATCG"),
                                "expected": False,
                                "desc": "Length Mismatch (Deletion)",
                            },
                            {
                                "input": ("ATATAT", "ATATAC"),
                                "expected": False,
                                "desc": "Mutation (T -> C)",
                            },
                            {
                                "input": ("", ""),
                                "expected": True,
                                "desc": "Empty Samples",
                            },
                            {
                                "input": ("A" * 1000 + "T", "T" + "A" * 1000),
                                "expected": True,
                                "desc": "Large Genome Stress Test",
                            },
                        ]

                        passed_count = 0
                        all_passed = True

                        st.write("### 🔬 Test Results")

                        for i, case in enumerate(test_cases):
                            ref, pat = case["input"]
                            expected = case["expected"]
                            try:
                                result = func(ref, pat)
                                if result == expected:
                                    st.markdown(
                                        f"✅ **Test {i+1}: {case['desc']}** — Passed"
                                    )
                                    passed_count += 1
                                else:
                                    st.markdown(
                                        f"❌ **Test {i+1}: {case['desc']}** — FAILED"
                                    )
                                    st.markdown(
                                        f"&nbsp;&nbsp;&nbsp;&nbsp;*Input: {ref}, {pat} | Expected: {expected} | Got: {result}*"
                                    )
                                    all_passed = False
                            except Exception as e:
                                st.error(f"Test {i+1} caused an error: {e}")
                                all_passed = False

                        st.write("---")
                        if all_passed:
                            st.balloons()
                            st.markdown(
                                """
                            <div class="success-box">
                                <h3>🎉 Diagnosis Confirmed!</h3>
                                <p>Your algorithm correctly identified all viral mutations. Dr. Nila is impressed.</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                            <div class="fail-box">
                                <h3>⚠️ Clinical Error</h3>
                                <p>You passed {passed_count} out of {len(test_cases)} tests. Review your logic and try again.</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                except Exception as e:
                    st.error("Syntax Error in your code:")
                    st.code(traceback.format_exc())
# --- PROBLEM 2 DATA ---
elif problem_selection == "2. The RNA Folding Validator":
    st.title("2. The RNA Folding Validator")

    # Badges
    c1, c2, c3 = st.columns([1, 1, 8])
    with c1:
        st.markdown("![Medium](https://img.shields.io/badge/Difficulty-Medium-orange)")
    with c2:
        st.markdown("![Topic](https://img.shields.io/badge/Topic-Stack-blue)")

    st.write("---")

    col_desc, col_code = st.columns([1, 1])

    with col_desc:
        st.subheader("🧬 The Clinical Scenario")
        st.write(
            """
        You are analyzing synthetic RNA sequences for a vaccine. 
        RNA strands fold into complex 3D shapes. This folding is represented by brackets:
        * `(` bonds with `)`
        * `{` bonds with `}`
        * `[` bonds with `]`
        
        For an RNA molecule to be stable, every "opening" base must have a corresponding "closing" base in the correct order. 
        
        **Your Task:**
        Write a function `is_stable_rna(structure)` that returns `True` if the structure is valid (chemically stable) and `False` if it is unstable (mismatched brackets).
        """
        )

        st.info(
            """
        **Example 1 (Stable Hairpin):**
        \nInput: structure = "({[]})"
        \nOutput: True
        """
        )

        st.info(
            """
        **Example 2 (Unstable/Broken):**
        \nInput: structure = "([)]"
        \nOutput: False
        \n(Explanation: The `[` was closed by `)` which is chemically impossible.)
        """
        )

    with col_code:
        st.subheader("💻 Your Solution")
        st.markdown("Use a **Stack** data structure to solve this.")

        default_code = """def is_stable_rna(structure):
    # Hint: Use a list as a stack.
    # push opening brackets, pop checking closing brackets.
    
    stack = []
    
    # Write your logic here...
    
    return True"""

        user_code = st.text_area(
            "Code Editor", value=default_code, height=350, key="p2_editor"
        )

        if st.button("Run Stability Test", type="primary"):
            with st.spinner("Simulating molecular folding..."):
                time.sleep(1)

                try:
                    local_scope = {}
                    exec(user_code, {}, local_scope)

                    if "is_stable_rna" not in local_scope:
                        st.error("Error: Function 'is_stable_rna' not found.")
                    else:
                        func = local_scope["is_stable_rna"]

                        # --- HIDDEN TEST CASES (Valid Parentheses) ---
                        test_cases = [
                            {"input": "()", "expected": True, "desc": "Simple Pair"},
                            {
                                "input": "()[]{}",
                                "expected": True,
                                "desc": "Multiple Domains",
                            },
                            {
                                "input": "(]",
                                "expected": False,
                                "desc": "Mismatched Bond",
                            },
                            {
                                "input": "([)]",
                                "expected": False,
                                "desc": "Interleaved (Impossible Fold)",
                            },
                            {
                                "input": "{[]}",
                                "expected": True,
                                "desc": "Nested Structure",
                            },
                            {
                                "input": "]",
                                "expected": False,
                                "desc": "Lonely Closing Bracket",
                            },
                        ]

                        passed_count = 0
                        all_passed = True

                        st.write("### 🔬 Test Results")

                        for i, case in enumerate(test_cases):
                            inp = case["input"]
                            expected = case["expected"]
                            try:
                                result = func(inp)
                                if result == expected:
                                    st.markdown(
                                        f"✅ **Test {i+1}: {case['desc']}** — Passed"
                                    )
                                    passed_count += 1
                                else:
                                    st.markdown(
                                        f"❌ **Test {i+1}: {case['desc']}** — FAILED"
                                    )
                                    st.markdown(
                                        f"&nbsp;&nbsp;&nbsp;&nbsp;*Input: '{inp}' | Expected: {expected} | Got: {result}*"
                                    )
                                    all_passed = False
                            except Exception as e:
                                st.error(f"Test {i+1} Error: {e}")
                                all_passed = False

                        if all_passed:
                            st.balloons()
                            st.success(
                                "Structure Stable! The vaccine candidate is viable."
                            )
                except Exception as e:
                    st.error("Syntax Error:")
                    st.code(traceback.format_exc())

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.caption("Created by Kuzhali's Student • 2026")
