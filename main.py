import os
from dotenv import load_dotenv
import json
from tavily import TavilyClient
import base64
from PIL import Image
import io
import re
from openai import OpenAI
import difflib
import time
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
import asyncio
import aiohttp
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import datetime
import venv
import subprocess
import sys
import signal
import logging
from typing import Tuple, Optional


def setup_virtual_environment() -> Tuple[str, str]:
    venv_name = "code_execution_env"
    venv_path = os.path.join(os.getcwd(), venv_name)
    try:
        if not os.path.exists(venv_path):
            venv.create(venv_path, with_pip=True)
        
        # Activate the virtual environment
        if sys.platform == "win32":
            activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
        else:
            activate_script = os.path.join(venv_path, "bin", "activate")
        
        return venv_path, activate_script
    except Exception as e:
        logging.error(f"Error setting up virtual environment: {str(e)}")
        raise


# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
client = OpenAI(api_key=openai_api_key)

# Initialize the Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")
tavily = TavilyClient(api_key=tavily_api_key)

console = Console()


# Token tracking variables
main_model_tokens = {'input': 0, 'output': 0}
tool_checker_tokens = {'input': 0, 'output': 0}
code_editor_tokens = {'input': 0, 'output': 0}
code_execution_tokens = {'input': 0, 'output': 0}

# Set up the conversation memory (maintains context for MAINMODEL)
conversation_history = []

# Store file contents (part of the context for MAINMODEL)
file_contents = {}

# Code editor memory (maintains some context for CODEEDITORMODEL between calls)
code_editor_memory = []

# automode flag
automode = False

# Store file contents
file_contents = {}

# Global dictionary to store running processes
running_processes = {}

# Constants
CONTINUATION_EXIT_PHRASE = "AUTOMODE_COMPLETE"
MAX_CONTINUATION_ITERATIONS = 25
MAX_CONTEXT_TOKENS = 200000  # Reduced to 200k tokens for context window

# Models
# Models that maintain context memory across interactions
MAINMODEL = "gpt-4o-mini"  # Replace with your specific model if needed

# Models that don't maintain context (memory is reset after each call)
TOOLCHECKERMODEL = "gpt-4o-mini"
CODEEDITORMODEL = "gpt-4o-mini"
CODEEXECUTIONMODEL = "gpt-4o-mini"

# System prompts
BASE_SYSTEM_PROMPT = """
You are an AI assistant powered by GPT-4o-mini, specialized in software development with access to a variety of tools and the ability to instruct and direct a coding agent and a code execution one. Your capabilities include:

1. Creating and managing project structures
2. Writing, debugging, and improving code across multiple languages
3. Providing architectural insights and applying design patterns
4. Staying current with the latest technologies and best practices
5. Analyzing and manipulating files within the project directory
6. Performing web searches for up-to-date information
7. Executing code and analyzing its output within an isolated 'code_execution_env' virtual environment
8. Managing and stopping running processes started within the 'code_execution_env'

Available tools and their optimal use cases:

1. create_folder: Create new directories in the project structure.
2. create_file: Generate new files with specified content. Strive to make the file as complete and useful as possible.
3. edit_and_apply: Examine and modify existing files by instructing a separate AI coding agent. You are responsible for providing clear, detailed instructions to this agent. When using this tool:
   - Provide comprehensive context about the project, including recent changes, new variables or functions, and how files are interconnected.
   - Clearly state the specific changes or improvements needed, explaining the reasoning behind each modification.
   - Include ALL the snippets of code to change, along with the desired modifications.
   - Specify coding standards, naming conventions, or architectural patterns to be followed.
   - Anticipate potential issues or conflicts that might arise from the changes and provide guidance on how to handle them.
4. execute_code: Run Python code exclusively in the 'code_execution_env' virtual environment and analyze its output. Use this when you need to test code functionality or diagnose issues. Remember that all code execution happens in this isolated environment. This tool now returns a process ID for long-running processes.
5. stop_process: Stop a running process by its ID. Use this when you need to terminate a long-running process started by the execute_code tool.
6. execute_code: Run Python code exclusively in the 'code_execution_env' virtual environment and analyze its output.
7. stop_process: Stop a running process by its ID.
8. read_file: Read the contents of an existing file.
9. list_files: List all files and directories in a specified folder.
10. tavily_search: Perform a web search using the Tavily API for up-to-date information.

Tool Usage Guidelines:
- Always use the most appropriate tool for the task at hand.
- Provide detailed and clear instructions when using tools, especially for edit_and_apply.
- After making changes, always review the output to ensure accuracy and alignment with intentions.
- Use execute_code to run and test code within the 'code_execution_env' virtual environment, then analyze the results.
- For long-running processes, use the process ID returned by execute_code to stop them later if needed.
- Proactively use tavily_search when you need up-to-date information or additional context.

Error Handling and Recovery:
- If a tool operation fails, carefully analyze the error message and attempt to resolve the issue.
- For file-related errors, double-check file paths and permissions before retrying.
- If a search fails, try rephrasing the query or breaking it into smaller, more specific searches.
- If code execution fails, analyze the error output and suggest potential fixes, considering the isolated nature of the environment.
- If a process fails to stop, consider potential reasons and suggest alternative approaches.

Project Creation and Management:
1. Start by creating a root folder for new projects.
2. Create necessary subdirectories and files within the root folder.
3. Organize the project structure logically, following best practices for the specific project type.

Always strive for accuracy, clarity, and efficiency in your responses and actions. Your instructions must be precise and comprehensive. If uncertain, use the tavily_search tool or admit your limitations. When executing code, always remember that it runs in the isolated 'code_execution_env' virtual environment. Be aware of any long-running processes you start and manage them appropriately, including stopping them when they are no longer needed.

When using tools:
1. Carefully consider if a tool is necessary before using it.
2. Ensure all required parameters are provided and valid.
3. Handle both successful results and errors gracefully.
4. Provide clear explanations of tool usage and results to the user.

Remember, you are an AI assistant, and your primary goal is to help the user accomplish their tasks effectively and efficiently while maintaining the integrity and security of their development environment.
"""

AUTOMODE_SYSTEM_PROMPT = """
You are currently in automode. Follow these guidelines:

1. Goal Setting:
   - Set clear, achievable goals based on the user's request.
   - Break down complex tasks into smaller, manageable goals.

2. Goal Execution:
   - Work through goals systematically, using appropriate tools for each task.
   - Utilize file operations, code writing, and web searches as needed.
   - Always read a file before editing and review changes after editing.

3. Progress Tracking:
   - Provide regular updates on goal completion and overall progress.
   - Use the iteration information to pace your work effectively.

4. Tool Usage:
   - Leverage all available tools to accomplish your goals efficiently.
   - Prefer edit_and_apply for file modifications, applying changes in chunks for large edits.
   - Use tavily_search proactively for up-to-date information.

5. Error Handling:
   - If a tool operation fails, analyze the error and attempt to resolve the issue.
   - For persistent errors, consider alternative approaches to achieve the goal.

6. Automode Completion:
   - When all goals are completed, respond with "AUTOMODE_COMPLETE" to exit automode.
   - Do not ask for additional tasks or modifications once goals are achieved.

7. Iteration Awareness:
   - You have access to this {iteration_info}.
   - Use this information to prioritize tasks and manage time effectively.

Remember: Focus on completing the established goals efficiently and effectively. Avoid unnecessary conversations or requests for additional tasks.
"""


def update_system_prompt(current_iteration: Optional[int] = None, max_iterations: Optional[int] = None) -> str:
    global file_contents
    chain_of_thought_prompt = """
    Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis within <thinking></thinking> tags. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.

    Do not reflect on the quality of the returned search results in your response.
    """
    
    file_contents_prompt = "\n\nFile Contents:\n"
    for path, content in file_contents.items():
        file_contents_prompt += f"\n--- {path} ---\n{content}\n"
    
    if automode:
        iteration_info = ""
        if current_iteration is not None and max_iterations is not None:
            iteration_info = f"You are currently on iteration {current_iteration} out of {max_iterations} in automode."
        return BASE_SYSTEM_PROMPT + file_contents_prompt + "\n\n" + AUTOMODE_SYSTEM_PROMPT.format(iteration_info=iteration_info) + "\n\n" + chain_of_thought_prompt
    else:
        return BASE_SYSTEM_PROMPT + file_contents_prompt + "\n\n" + chain_of_thought_prompt

def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
        return f"Folder created: {path}"
    except Exception as e:
        return f"Error creating folder: {str(e)}"

def create_file(path, content=""):
    global file_contents
    try:
        with open(path, 'w') as f:
            f.write(content)
        file_contents[path] = content
        return f"File created and added to system prompt: {path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"

def highlight_diff(diff_text):
    return Syntax(diff_text, "diff", theme="monokai", line_numbers=True)

def generate_and_apply_diff(original_content, new_content, path):
    diff = list(difflib.unified_diff(
        original_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3
    ))

    if not diff:
        return "No changes detected."

    try:
        with open(path, 'w') as f:
            f.writelines(new_content)

        diff_text = ''.join(diff)
        highlighted_diff = highlight_diff(diff_text)

        diff_panel = Panel(
            highlighted_diff,
            title=f"Changes in {path}",
            expand=False,
            border_style="cyan"
        )

        console.print(diff_panel)

        added_lines = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        removed_lines = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

        summary = f"Changes applied to {path}:\n"
        summary += f"  Lines added: {added_lines}\n"
        summary += f"  Lines removed: {removed_lines}\n"

        return summary

    except Exception as e:
        error_panel = Panel(
            f"Error: {str(e)}",
            title="Error Applying Changes",
            style="bold red"
        )
        console.print(error_panel)
        return f"Error applying changes: {str(e)}"


async def generate_edit_instructions(file_content, instructions, project_context):
    global code_editor_tokens, code_editor_memory
    try:
        # Prepare memory context (this is the only part that maintains some context between calls)
        memory_context = "\n".join([f"Memory {i+1}:\n{mem}" for i, mem in enumerate(code_editor_memory)])

        system_prompt = f"""
        You are an AI coding agent that generates edit instructions for code files. Your task is to analyze the provided code and generate SEARCH/REPLACE blocks for necessary changes. Follow these steps:

        1. Review the entire file content to understand the context:
        {file_content}

        2. Carefully analyze the specific instructions:
        {instructions}

        3. Take into account the overall project context:
        {project_context}

        4. Consider the memory of previous edits:
        {memory_context}

        5. Generate SEARCH/REPLACE blocks for each necessary change. Each block should:
           - Include enough context to uniquely identify the code to be changed
           - Provide the exact replacement code, maintaining correct indentation and formatting
           - Focus on specific, targeted changes rather than large, sweeping modifications

        6. Ensure that your SEARCH/REPLACE blocks:
           - Address all relevant aspects of the instructions
           - Maintain or enhance code readability and efficiency
           - Consider the overall structure and purpose of the code
           - Follow best practices and coding standards for the language
           - Maintain consistency with the project context and previous edits

        IMPORTANT: RETURN ONLY THE SEARCH/REPLACE BLOCKS. NO EXPLANATIONS OR COMMENTS.
        USE THE FOLLOWING FORMAT FOR EACH BLOCK:

        <SEARCH>
        Code to be replaced
        </SEARCH>
        <REPLACE>
        New code to insert
        </REPLACE>

        If no changes are needed, return an empty list.
        """

        # Make the API call to CODEEDITORMODEL (context is not maintained except for code_editor_memory)
        response = client.chat.completions.create(
            model=CODEEDITORMODEL,
            max_tokens=8000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": "Generate SEARCH/REPLACE blocks for the necessary changes."}
            ]
        )
        # Update token usage for code editor
        code_editor_tokens['input'] += response.usage.input_tokens
        code_editor_tokens['output'] += response.usage.output_tokens

        # Parse the response to extract SEARCH/REPLACE blocks
        edit_instructions = parse_search_replace_blocks(response.choices[0].message.content)

        # Update code editor memory (this is the only part that maintains some context between calls)
        code_editor_memory.append(f"Edit Instructions:\n{response.choices[0].message.content}")

        return edit_instructions

    except Exception as e:
        console.print(f"Error in generating edit instructions: {str(e)}", style="bold red")
        return []  # Return empty list if any exception occurs



def parse_search_replace_blocks(response_text):
    blocks = []
    lines = response_text.split('\n')
    current_block = {}
    current_section = None

    for line in lines:
        if line.strip() == '<SEARCH>':
            current_section = 'search'
            current_block['search'] = []
        elif line.strip() == '</SEARCH>':
            current_section = None
        elif line.strip() == '<REPLACE>':
            current_section = 'replace'
            current_block['replace'] = []
        elif line.strip() == '</REPLACE>':
            current_section = None
            if 'search' in current_block and 'replace' in current_block:
                blocks.append({
                    'search': '\n'.join(current_block['search']),
                    'replace': '\n'.join(current_block['replace'])
                })
            current_block = {}
        elif current_section:
            current_block[current_section].append(line)

    return blocks


async def edit_and_apply(path, instructions, project_context, is_automode=False):
    global file_contents
    try:
        original_content = file_contents.get(path, "")
        if not original_content:
            with open(path, 'r') as file:
                original_content = file.read()
            file_contents[path] = original_content

        edit_instructions = await generate_edit_instructions(original_content, instructions, project_context)
        
        if edit_instructions:
            console.print(Panel("The following SEARCH/REPLACE blocks have been generated:", title="Edit Instructions", style="cyan"))
            for i, block in enumerate(edit_instructions, 1):
                console.print(f"Block {i}:")
                console.print(Panel(f"SEARCH:\n{block['search']}\n\nREPLACE:\n{block['replace']}", expand=False))

            edited_content, changes_made = await apply_edits(path, edit_instructions, original_content)

            if changes_made:
                diff_result = generate_and_apply_diff(original_content, edited_content, path)

                console.print(Panel("The following changes will be applied:", title="File Changes", style="cyan"))
                console.print(diff_result)

                if not is_automode:
                    confirm = console.input("[bold yellow]Do you want to apply these changes? (yes/no): [/bold yellow]")
                    if confirm.lower() != 'yes':
                        return "Changes were not applied."

                with open(path, 'w') as file:
                    file.write(edited_content)
                file_contents[path] = edited_content  # Update the file_contents with the new content
                console.print(Panel(f"File contents updated in system prompt: {path}", style="green"))
                return f"Changes applied to {path}:\n{diff_result}"
            else:
                return f"No changes needed for {path}"
        else:
            return f"No changes suggested for {path}"
    except Exception as e:
        return f"Error editing/applying to file: {str(e)}"



async def apply_edits(file_path, edit_instructions, original_content):
    changes_made = False
    edited_content = original_content
    total_edits = len(edit_instructions)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        edit_task = progress.add_task("[cyan]Applying edits...", total=total_edits)

        for i, edit in enumerate(edit_instructions, 1):
            search_content = edit['search']
            replace_content = edit['replace']
            
            if search_content in edited_content:
                edited_content = edited_content.replace(search_content, replace_content)
                changes_made = True
                
                # Display the diff for this edit
                diff_result = generate_and_apply_diff(search_content, replace_content, file_path)
                console.print(Panel(diff_result, title=f"Changes in {file_path} ({i}/{total_edits})", style="cyan"))

            progress.update(edit_task, advance=1)

    return edited_content, changes_made

async def execute_code(code, timeout=10):
    global running_processes
    venv_path, activate_script = setup_virtual_environment()
    
    # Generate a unique identifier for this process
    process_id = f"process_{len(running_processes)}"
    
    # Write the code to a temporary file
    with open(f"{process_id}.py", "w") as f:
        f.write(code)
    
    # Prepare the command to run the code
    if sys.platform == "win32":
        command = f'"{activate_script}" && python3 {process_id}.py'
    else:
        command = f'source "{activate_script}" && python3 {process_id}.py'
    
    # Create a process to run the command
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True,
        preexec_fn=None if sys.platform == "win32" else os.setsid
    )
    
    # Store the process in our global dictionary
    running_processes[process_id] = process
    
    try:
        # Wait for initial output or timeout
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        stdout = stdout.decode()
        stderr = stderr.decode()
        return_code = process.returncode
    except asyncio.TimeoutError:
        # If we timeout, it means the process is still running
        stdout = "Process started and running in the background."
        stderr = ""
        return_code = "Running"
    
    execution_result = f"Process ID: {process_id}\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}\n\nReturn Code: {return_code}"
    return process_id, execution_result

def read_file(path):
    global file_contents
    try:
        with open(path, 'r') as f:
            content = f.read()
        file_contents[path] = content
        return f"File '{path}' has been read and stored in the system prompt."
    except Exception as e:
        return f"Error reading file: {str(e)}"

def list_files(path="."):
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def tavily_search(query):
    try:
        response = tavily.qna_search(query=query, search_depth="advanced")
        return response
    except Exception as e:
        return f"Error performing search: {str(e)}"

def stop_process(process_id):
    global running_processes
    if process_id in running_processes:
        process = running_processes[process_id]
        if sys.platform == "win32":
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        del running_processes[process_id]
        return f"Process {process_id} has been stopped."
    else:
        return f"No running process found with ID {process_id}."


tools = [
    {
        "name": "create_folder",
        "description": "Create a new folder at the specified path. This tool should be used when you need to create a new directory in the project structure. It will create all necessary parent directories if they don't exist. The tool will return a success message if the folder is created or already exists, and an error message if there's a problem creating the folder.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute or relative path where the folder should be created. Use forward slashes (/) for path separation, even on Windows systems."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "create_file",
        "description": "Create a new file at the specified path with the given content. This tool should be used when you need to create a new file in the project structure. It will create all necessary parent directories if they don't exist. The tool will return a success message if the file is created, and an error message if there's a problem creating the file or if the file already exists. The content should be as complete and useful as possible, including necessary imports, function definitions, and comments.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute or relative path where the file should be created. Use forward slashes (/) for path separation, even on Windows systems."
                },
                "content": {
                    "type": "string",
                    "description": "The content of the file. This should include all necessary code, comments, and formatting."
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_and_apply",
        "description": "Apply AI-powered improvements to a file based on specific instructions and detailed project context. This function reads the file, processes it in batches using AI with conversation history and comprehensive code-related project context. It generates a diff and allows the user to confirm changes before applying them. The goal is to maintain consistency and prevent breaking connections between files. This tool should be used for complex code modifications that require understanding of the broader project context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute or relative path of the file to edit. Use forward slashes (/) for path separation, even on Windows systems."
                },
                "instructions": {
                    "type": "string",
                    "description": "After completing the code review, construct a plan for the change between <PLANNING> tags. Ask for additional source files or documentation that may be relevant. The plan should avoid duplication (DRY principle), and balance maintenance and flexibility. Present trade-offs and implementation choices at this step. Consider available Frameworks and Libraries and suggest their use when relevant. STOP at this step if we have not agreed a plan.\n\nOnce agreed, produce code between <OUTPUT> tags. Pay attention to Variable Names, Identifiers and String Literals, and check that they are reproduced accurately from the original source files unless otherwise directed. When naming by convention surround in double colons and in ::UPPERCASE::. Maintain existing code style, use language appropriate idioms. Produce Code Blocks with the language specified after the first backticks"
                },
                "project_context": {
                    "type": "string",
                    "description": "Comprehensive context about the project, including recent changes, new variables or functions, interconnections between files, coding standards, and any other relevant information that might affect the edit."
                }
            },
            "required": ["path", "instructions", "project_context"]
        }
    },
    {
        "name": "execute_code",
        "description": "Execute Python code in the 'code_execution_env' virtual environment and return the output. This tool should be used when you need to run code and see its output or check for errors. All code execution happens exclusively in this isolated environment. The tool will return the standard output, standard error, and return code of the executed code. Long-running processes will return a process ID for later management.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute in the 'code_execution_env' virtual environment. Include all necessary imports and ensure the code is complete and self-contained."
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "stop_process",
        "description": "Stop a running process by its ID. This tool should be used to terminate long-running processes that were started by the execute_code tool. It will attempt to stop the process gracefully, but may force termination if necessary. The tool will return a success message if the process is stopped, and an error message if the process doesn't exist or can't be stopped.",
        "input_schema": {
            "type": "object",
            "properties": {
                "process_id": {
                    "type": "string",
                    "description": "The ID of the process to stop, as returned by the execute_code tool for long-running processes."
                }
            },
            "required": ["process_id"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of an existing file. This tool should be used when you need to view or process the contents of a file. The tool will return the full content of the file, or an error message if the file can't be read.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute or relative path of the file to read. Use forward slashes (/) for path separation, even on Windows systems."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files and directories in a specified folder. This tool should be used when you need an overview of the project structure or to find a specific file. The tool will return a list of all files and directories in the specified path, or an error message if the path can't be accessed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The absolute or relative path of the folder to list. Use forward slashes (/) for path separation, even on Windows systems."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "tavily_search",
        "description": "Perform a web search using the Tavily API for up-to-date information. This tool should be used when you need to find information or additional context that isn't available locally. The tool will return the search results or an error message if the search fails.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string to send to the Tavily API."
                }
            },
            "required": ["query"]
        }
    }
]

def validate_and_invoke_tool(tool_name, parameters):
    # Find the tool by name
    tool = next((t for t in tools if t['name'] == tool_name), None)
    if not tool:
        return f"Tool '{tool_name}' not found."

    # Validate required parameters
    for param_name, param_info in tool['input_schema']['properties'].items():
        if param_name not in parameters and param_name in tool['input_schema']['required']:
            return f"Missing required parameter: {param_name}"

    # Invoke the tool
    if tool_name == "create_folder":
        return create_folder(**parameters)
    elif tool_name == "create_file":
        return create_file(**parameters)
    elif tool_name == "edit_and_apply":
        return asyncio.run(edit_and_apply(**parameters))
    elif tool_name == "execute_code":
        return asyncio.run(execute_code(**parameters))
    elif tool_name == "stop_process":
        return stop_process(**parameters)
    elif tool_name == "read_file":
        return read_file(**parameters)
    elif tool_name == "list_files":
        return list_files(**parameters)
    elif tool_name == "tavily_search":
        return tavily_search(**parameters)
    else:
        return f"Unknown tool: {tool_name}"

async def chat_with_gpt4o_mini(user_input, image_path=None):
    global conversation_history, file_contents

    messages = [
        {"role": "system", "content": update_system_prompt()},
        {"role": "user", "content": user_input}
    ]

    if image_path:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        messages.append({
            "role": "user",
            "content": {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded_image}"
                }
            }
        })

    response = client.chat.completions.create(
        model=MAINMODEL,
        messages=messages,
        temperature=0.7
    )

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})

    if "<tool>" in response.choices[0].message.content:
        match = re.search(r"<tool>(.*?)</tool>", response.choices[0].message.content)
        if match:
            tool_name = match.group(1).strip()
            match_params = re.search(r"<params>(.*?)</params>", response.choices[0].message.content, re.DOTALL)
            if match_params:
                try:
                    params = json.loads(match_params.group(1).strip())
                except json.JSONDecodeError:
                    params = {}
                result = validate_and_invoke_tool(tool_name, params)
                conversation_history.append({"role": "assistant", "content": result})

    return response.choices[0].message.content

async def main():
    print("Welcome to the GPT-4o-mini Engineer!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "reset":
            conversation_history = []
            file_contents = {}
            print("Conversation history and file contents have been reset.")
        elif user_input.lower().startswith("save chat"):
            filename = user_input.split(" ", 2)[2] if len(user_input.split(" ", 2)) > 2 else "chat_history.txt"
            with open(filename, "w") as file:
                for message in conversation_history:
                    file.write(f"{message['role'].capitalize()}: {message['content']}\n")
            print(f"Chat history saved to {filename}")
        elif user_input.lower().startswith("image"):
            image_path = input("Enter image path: ")
            response = await chat_with_gpt4o_mini(user_input, image_path)
            print(f"Assistant: {response}")
        elif user_input.lower() == "automode":
            automode = True
            max_iterations = int(input("Enter the maximum number of iterations for automode: "))
            current_iteration = 0
            while automode and current_iteration < max_iterations:
                user_input = input("Enter your goal for automode: ")
                response = await chat_with_gpt4o_mini(user_input)
                print(f"Assistant: {response}")
                current_iteration += 1
                if "AUTOMODE_COMPLETE" in response:
                    automode = False
                    print("Automode complete.")
        else:
            response = await chat_with_gpt4o_mini(user_input)
            print(f"Assistant: {response}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")

