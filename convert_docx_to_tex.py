
import docx
import re
import os

def escape_latex(text):
    """
    Escapes special characters for LaTeX.
    """
    if not text:
        return ""
    # Order matters: backslash first
    text = text.replace('\\', r'\textbackslash{}')
    
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    # Don't replace backslash again
    for k, v in replacements.items():
        text = text.replace(k, v)
        
    return text

def process_run(run):
    """
    Process a run to handle bold/italic formatting.
    """
    text = escape_latex(run.text)
    if not text:
        return ""
        
    if run.bold:
        text = f"\\textbf{{{text}}}"
    if run.italic:
        text = f"\\textit{{{text}}}"
    return text

def get_para_content(para):
    content = ""
    for run in para.runs:
        content += process_run(run)
    return content

def convert_docx_to_tex(docx_path, tex_path):
    doc = docx.Document(docx_path)
    
    lines = []
    
    # Preamble
    lines.append(r'\documentclass{article}')
    lines.append(r'\usepackage[utf8]{inputenc}')
    lines.append(r'\usepackage{geometry}')
    lines.append(r'\geometry{a4paper, margin=1in}')
    lines.append(r'\usepackage{graphicx}')
    lines.append(r'\usepackage{hyperref}')
    lines.append(r'\usepackage{xeCJK}') # For Chinese support
    # Try to set a font if on a system that might have it, or rely on default
    lines.append(r'% \setCJKmainfont{SimSun} % Uncomment if needed') 
    lines.append(r'\title{Converted Document}')
    lines.append(r'\author{}')
    lines.append(r'\date{\today}')
    lines.append(r'\begin{document}')
    lines.append(r'\maketitle')
    lines.append(r'')

    in_list = False
    list_type = None # 'itemize' or 'enumerate'

    for para in doc.paragraphs:
        style_name = para.style.name
        content = get_para_content(para)
        
        if not content.strip():
            continue

        # Check if we need to close a list
        if in_list and 'List' not in style_name:
            lines.append(f"\\end{{{list_type}}}")
            in_list = False
            list_type = None

        if 'Heading 1' in style_name:
            lines.append(f"\\section{{{content}}}")
        elif 'Heading 2' in style_name:
            lines.append(f"\\subsection{{{content}}}")
        elif 'Heading 3' in style_name:
            # Shift up if no H1/H2 found, or keep strict? 
            # Given styles found: Heading 3, Heading 4. Let's treat Heading 3 as Section for now.
            lines.append(f"\\section{{{content}}}") 
        elif 'Heading 4' in style_name:
            lines.append(f"\\subsection{{{content}}}")
        elif 'List Paragraph' in style_name:
            if not in_list:
                in_list = True
                list_type = 'itemize' # Default to itemize for List Paragraph
                lines.append(f"\\begin{{{list_type}}}")
            lines.append(f"  \\item {content}")
        else:
            # Normal text
            lines.append(f"{content}\n")
            
    if in_list:
        lines.append(f"\\end{{{list_type}}}")
            
    lines.append(r'\end{document}')
    
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Successfully converted {docx_path} to {tex_path}")

if __name__ == "__main__":
    docx_file = "论文.docx"
    tex_file = "论文.tex"
    if os.path.exists(docx_file):
        convert_docx_to_tex(docx_file, tex_file)
    else:
        print(f"Error: {docx_file} not found.")
