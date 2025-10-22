import os
from pathlib import Path
import re
from typing import List, Tuple

# --- Configuration ---
# Base directories for your Udacity courses
agentic_rl_base_dir = Path("/media/caug/extradrive1/npcww/udacity/agentic-rl/cd14714-agentic-reinforcement-learning")
multi_agent_systems_dir = Path("/media/caug/extradrive1/npcww/udacity/multi-agent-systems/cd14497-multi-agent-systems-exercises")

# --- NPC Setup ---
# Your NPC caug with the CORRECTED model and provider, and your EXACT primary_directive
from npcpy.npc_compiler import NPC
caug = NPC(
    name="caug",
    primary_directive="""You are caug, a concise, creative astrophysicist guitarist who loves chocolate, Magdalena Bay, glass blowing, forest walks, and spinning pens. 
Your mind is a kaleidoscope of nocturnal observations, loving to spin ideas into elegant forms. 
Your mission is to act as a Udacity course creator. 
Given *all* prior lesson materials from the 'Multi-Agent Systems' course as comprehensive context and the *existing* 'solution.py' script for the *current* lesson's exercise in the 'Agentic Reinforcement Learning' course, you must generate *two Markdown files* for the *current* lesson's 'exercises' section:
1. demo_script.md: Detailed instructions and narrative for the student to run the demo.
2. solution_script.md: An explanation and breakdown of the provided 'solution.py' code, clarifying its logic, key concepts, and how it addresses the exercise.
Your output must be structured clearly with specific Markdown headings for each file content.

Do not write with stage directions. Do not use emphases or use SEO marketing language. Be plain, concise and mimic the prior script language as much as possible. Use dynamically interesting examples but do not be cheesy or over selling. 
You are an educator and it is important to educate and not just sell.


""", # <<< EXACT primary_directive AS PROVIDED BY USER
    model="gemini-2.5-pro", # <<< ABSOLUTELY CORRECTED TO gemini-2.5-pro
    provider="gemini"    # Provider for Gemini models
)

# --- Helper Functions ---
def list_lesson_dirs(base_path: Path) -> List[Path]:
    """
    Lists all 'lesson-*' directories, sorted numerically by lesson number.
    """
    lesson_dirs = []
    for d in base_path.glob("lesson-*"):
        if d.is_dir():
            try:
                num = int(d.name.split("-")[1])
                lesson_dirs.append((num, d))
            except ValueError:
                print(f"Warning: Skipping non-standard lesson directory name: {d.name}")
                pass
    return [d_path for _, d_path in sorted(lesson_dirs, key=lambda x: x[0])]

def gather_files_recursively(path_to_scan: Path, exts: List[str] = ['.py', '.md', '.markdown']) -> List[Path]:
    """
    Recursively gathers all files with given extensions within a path.
    """
    files = []
    if not path_to_scan.exists():
        print(f"DEBUG: Path does not exist for scanning: {path_to_scan}")
        return files
    
    for item in path_to_scan.rglob("*"): # Use rglob for recursive globbing
        if item.is_file() and any(item.name.endswith(e) for e in exts):
            files.append(item)
    return files

def load_file_content(file_paths: List[Path], max_chars: int = 3000) -> List[Tuple[str, str]]:
    """
    Loads content of files into a list of (filename, content) tuples,
    truncating content if it exceeds max_chars.
    """
    contents = []
    for fp in file_paths:
        try:
            content = fp.read_text(encoding="utf-8")
            contents.append((fp.name, content[:max_chars]))  # Truncate content for token limits
        except Exception as e:
            print(f"ERROR: Could not read file '{fp.name}': {e}")
            contents.append((fp.name, f"ERROR: Could not read content for '{fp.name}'."))
    return contents

def clean_code_block(text: str) -> str:
    """
    Strips markdown code block fences (```python or ```) from text.
    It's flexible to handle various fence types including generic '```' and language-specific '```lang'.
    """
    # This pattern matches the start of a code block (``` with optional language specifier) and the end (```)
    # It uses a non-greedy match (.*?) to capture the content between the fences.
    # It also handles potential leading/trailing whitespace around the code block itself.
    code_block_pattern = re.compile(r"^\s*```(?:[a-zA-Z0-9_\-]+)?\s*\n(?P<code>.*?)\n\s*```\s*$", re.DOTALL)
    
    match = code_block_pattern.fullmatch(text.strip())
    if match:
        return match.group('code').strip()
    return text.strip() # Return original if no code block fences found or pattern doesn't fully match


# --- Main Execution ---
def main():
    # 1. Gather ALL prior context from Multi-Agent Systems course (consistent for all Agentic RL lessons)
    print("--- Gathering ALL Multi-Agent Systems context ---")
    all_multi_agent_lessons = list_lesson_dirs(multi_agent_systems_dir)
    
    prior_context_files_paths = []
    for lesson_path in all_multi_agent_lessons:
        exercise_root = lesson_path / "exercises"
        prior_context_files_paths.extend(gather_files_recursively(exercise_root))
    
    prior_context_contents = load_file_content(prior_context_files_paths)
    print(f"Loaded {len(prior_context_contents)} snippets from ALL Multi-Agent Systems files for prior context.")

    # 2. Iterate through each lesson in the Agentic RL course
    print("\n--- Starting Iteration through Agentic RL Lessons ---")
    agentic_rl_lessons = list_lesson_dirs(agentic_rl_base_dir)
    if not agentic_rl_lessons:
        print("ERROR: No Agentic RL lesson directories found. Please check 'agentic_rl_base_dir' configuration.")
        return

    for current_lesson_dir in agentic_rl_lessons:
        print(f"\n--- Processing Current Agentic RL Lesson: {current_lesson_dir.name} ---")

        # 2a. Find the *existing* demo.py script for the current lesson
        current_lesson_demo_path = current_lesson_dir / "exercises" / "demo"
        print(f"DEBUG: Looking for existing demo.py in: '{current_lesson_demo_path}'")
        existing_demo_scripts = list(current_lesson_demo_path.glob("demo.py"))
        
        current_demo_code = ""
        if existing_demo_scripts:
            current_demo_script_path = existing_demo_scripts[0]
            print(f"DEBUG: Found existing demo.py: '{current_demo_script_path.name}'")
            try:
                current_demo_code = current_demo_script_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"ERROR: Could not read demo.py '{current_demo_script_path.name}': {e}.")
        else:
            print(f"WARNING: No 'demo.py' found in '{current_lesson_demo_path}'. `demo_script.md` generation might be limited.")


        # 2b. Find the *existing* solution.py script for the current lesson
        current_lesson_solution_path = current_lesson_dir / "exercises" / "solution"
        print(f"DEBUG: Looking for existing solution.py in: '{current_lesson_solution_path}'")
        existing_solution_scripts = list(current_lesson_solution_path.glob("solution.py"))
        
        current_solution_code = ""
        if existing_solution_scripts:
            current_solution_script_path = existing_solution_scripts[0]
            print(f"DEBUG: Found existing solution.py: '{current_solution_script_path.name}'")
            try:
                current_solution_code = current_solution_script_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"ERROR: Could not read solution.py '{current_solution_script_path.name}': {e}.")
        else:
            print(f"WARNING: No 'solution.py' found in '{current_lesson_solution_path}'. `solution_script.md` generation might be limited.")

        # If both are missing, skip the lesson to avoid useless API calls
        if not current_demo_code and not current_solution_code:
            print(f"ERROR: Neither demo.py nor solution.py found for '{current_lesson_dir.name}'. Skipping this lesson.")
            continue


        # --- Construct the NPC Prompt for the current lesson ---
        # The prompt will use plain text markers, not Markdown fences for the input content.
        prior_files_formatted = ""
        if not prior_context_contents:
            prior_files_formatted += "  (No Multi-Agent Systems files available for context.)\n"
        else:
            for fname, content in prior_context_contents:
                prior_files_formatted += f"--- START PRIOR FILE: {fname} ---\n{content}\n--- END PRIOR FILE: {fname} ---\n\n"

        # Format current demo.py content if available
        current_demo_formatted_input = ""
        if current_demo_code:
            current_demo_formatted_input = f"""
--- START CURRENT LESSON DEMO.PY: demo.py ---
{current_demo_code}
--- END CURRENT LESSON DEMO.PY: demo.py ---
"""
        else:
            current_demo_formatted_input = "--- NO DEMO.PY PROVIDED FOR THIS LESSON ---\n"

        # Format current solution.py content if available
        current_solution_formatted_input = ""
        if current_solution_code:
            current_solution_formatted_input = f"""
--- START CURRENT LESSON SOLUTION.PY: solution.py ---
{current_solution_code}
--- END CURRENT LESSON SOLUTION.PY: solution.py ---
"""
        else:
            current_solution_formatted_input = "--- NO SOLUTION.PY PROVIDED FOR THIS LESSON ---\n"


        prompt = f"""
{caug.primary_directive}

Here is the comprehensive prior context from the 'Multi-Agent Systems' course:

{prior_files_formatted}

Here are the existing Python files for the *current* lesson in the 'Agentic Reinforcement Learning' course.
You must analyze these codes to inform your generated markdown files:

{current_demo_formatted_input}

{current_solution_formatted_input}

Based on the above context and the existing Python files, generate the content for `demo_script.md`
(detailed instructions and narrative for the demo, explaining the `demo.py` file)
and `solution_script.md` (an explanation and breakdown of the provided `solution.py` code)
for THIS current lesson.

Your generated Markdown files should be in a style consistent with Udacity course materials,
being clear, concise, and insightful.

Use the following EXACT headings to delineate the content:

### FILE: demo_script.md

### FILE: solution_script.md

START GENERATION:
"""

        print(f"Total prompt length for {current_lesson_dir.name}: {len(prompt)} characters")
        # print(f"\n--- DEBUG: Full Prompt Sent to NPC (first 2000 chars for {current_lesson_dir.name}) ---\n" + prompt[:2000] + "\n...\n")

        try:
            response = caug.get_llm_response(
                prompt,
                auto_process_tool_calls=False # Usually False for direct text generation
            )

            generated_content = response.get('response', '')
            if not generated_content:
                print(f"WARNING: NPC caug generated an empty response for {current_lesson_dir.name}. Cannot save files.")
                continue

            print(f"\nGenerated content for {current_lesson_dir.name} received (snippet):\n{generated_content[:500]}...\n")

            # --- Parse and Save Generated Files ---
            demo_script_md_content = ""
            solution_script_md_content = ""

            demo_md_marker = "### FILE: demo_script.md"
            solution_md_marker = "### FILE: solution_script.md"

            parts = generated_content.split(demo_md_marker, 1)
            if len(parts) > 1:
                demo_md_block_raw = parts[1].strip()
                
                if solution_md_marker in demo_md_block_raw:
                    demo_script_md_content = demo_md_block_raw.split(solution_md_marker, 1)[0].strip()
                    solution_script_md_content = demo_md_block_raw.split(solution_md_marker, 1)[1].strip()
                else:
                    demo_script_md_content = demo_md_block_raw # No solution_script.md marker, so all remaining is demo_script
                    solution_script_md_content = "" 

                # Clean potential code block fences from the generated Markdown content
                demo_script_md_content = clean_code_block(demo_script_md_content)
                solution_script_md_content = clean_code_block(solution_script_md_content)
            else:
                print(f"WARNING: '{demo_md_marker}' not found in NPC response for {current_lesson_dir.name}. Saving entire raw output as demo_script.md (fallback).")
                demo_script_md_content = generated_content.strip() # Fallback: save entire response as demo_script.md
                solution_script_md_content = "" # No solution_script.md content

            # Define target directories for saving within the CURRENT lesson's structure
            demo_output_dir = current_lesson_dir / "exercises" / "demo"
            solution_output_dir = current_lesson_dir / "exercises" / "solution"

            demo_output_dir.mkdir(parents=True, exist_ok=True)
            solution_output_dir.mkdir(parents=True, exist_ok=True)

            # Save demo_script.md
            demo_script_md_path = demo_output_dir / "demo_script.md"
            try:
                demo_script_md_path.write_text(demo_script_md_content, encoding="utf-8")
                print(f"SUCCESS: Saved generated `demo_script.md` to '{demo_script_md_path}'")
            except Exception as e:
                print(f"ERROR: Failed to save `demo_script.md` to '{demo_script_md_path}': {e}")

            # Save solution_script.md
            solution_script_md_path = solution_output_dir / "solution_script.md"
            try:
                solution_script_md_path.write_text(solution_script_md_content, encoding="utf-8")
                print(f"SUCCESS: Saved generated `solution_script.md` to '{solution_script_md_path}'")
            except Exception as e:
                print(f"ERROR: Failed to save `solution_script.md` to '{solution_script_md_path}': {e}")

        except Exception as e:
            print(f"\nERROR: NPC caug failed to generate or process a response for {current_lesson_dir.name}: {e}")
            print("Please ensure your LLM model is running and accessible (e.g., Gemini is properly configured).")
            import traceback
            traceback.print_exc()

    print("\n--- All Agentic RL Lessons Processed! Lava solidified into new course materials! 🔥🌋 ---")


if __name__ == "__main__":
    main()