# # src/response_formatter.py

# """
# Professional Response Formatter
# Converts RAG answers into beautifully formatted, well-structured responses
# with headings, subheadings, bold text, and proper visual hierarchy.
# """

# from typing import List, Dict
# import re

# class ResponseFormatter:
#     """Format RAG responses with professional structure and styling."""
    
#     def __init__(self):
#         self.markdown_enabled = True
    
#     @staticmethod
#     def format_answer_with_structure(
#         answer: str,
#         query: str,
#         citations: List[Dict],
#         confidence_level: str = "high"
#     ) -> str:
#         """
#         Format answer with professional structure.
        
#         Returns HTML-ready or Markdown formatted response.
#         """
        
#         # Break answer into sections
#         sections = ResponseFormatter._extract_sections(answer)
        
#         formatted = ""
        
#         # 1. MAIN HEADING
#         formatted += f"## 📋 Response to: \"{query}\"\n\n"
        
#         # 2. CONFIDENCE INDICATOR
#         confidence_icon = "✅ High" if confidence_level == "high" else "⚠️ Medium" if confidence_level == "medium" else "❌ Low"
#         formatted += f"**Confidence Level:** {confidence_icon}\n\n"
        
#         # 3. EXECUTIVE SUMMARY (if answer is long)
#         if len(answer) > 500:
#             summary = ResponseFormatter._extract_summary(answer)
#             formatted += f"### 📌 Summary\n{summary}\n\n"
        
#         # 4. DETAILED ANSWER (formatted with sections)
#         formatted += "### 📖 Detailed Answer\n\n"
        
#         if sections:
#             for section_title, section_content in sections:
#                 if section_title:
#                     formatted += f"#### {section_title}\n"
#                 formatted += f"{section_content}\n\n"
#         else:
#             formatted += f"{answer}\n\n"
        
#         # 5. KEY POINTS (extracted from answer)
#         key_points = ResponseFormatter._extract_key_points(answer)
#         if key_points:
#             formatted += "### 🎯 Key Points\n\n"
#             for i, point in enumerate(key_points, 1):
#                 formatted += f"{i}. {point}\n"
#             formatted += "\n"
        
#         # 6. SOURCES (formatted professionally)
#         formatted += ResponseFormatter._format_citations(citations)
        
#         return formatted
    
#     @staticmethod
#     def _extract_sections(text: str) -> List[tuple]:
#         """Extract sections from answer based on natural breaks."""
        
#         # Split on paragraph breaks (2+ newlines)
#         paragraphs = text.split('\n\n')
        
#         sections = []
        
#         for para in paragraphs:
#             if not para.strip():
#                 continue
            
#             # Check if paragraph starts with common section indicators
#             if para.strip().startswith(('Role of', 'In the', 'During', 'As the', 'The role of', 'One')):
#                 sections.append((None, para.strip()))
#             elif ':' in para[:50]:  # Title-like format
#                 title, content = para.split(':', 1)
#                 sections.append((title.strip(), content.strip()))
#             else:
#                 sections.append((None, para.strip()))
        
#         return sections
    
#     @staticmethod
#     def _extract_summary(text: str, sentences: int = 3) -> str:
#         """Extract summary (first few sentences)."""
        
#         # Get first N sentences
#         sentences_list = text.split('. ')
#         summary_sentences = sentences_list[:sentences]
        
#         summary = '. '.join(summary_sentences)
#         if not summary.endswith('.'):
#             summary += '.'
        
#         return summary
    
#     @staticmethod
#     def _extract_key_points(text: str) -> List[str]:
#         """Extract key points from answer."""
        
#         points = []
        
#         # Look for common key point patterns
#         sentences = text.split('. ')
        
#         for sentence in sentences:
#             sentence = sentence.strip()
            
#             # Include sentences with these patterns
#             if any(keyword in sentence.lower() for keyword in [
#                 'significant', 'important', 'role', 'responsibility', 'noted',
#                 'approved', 'decision', 'recommendation', 'suggest', 'contributed',
#                 'led to', 'resulted in', 'highlighted'
#             ]):
#                 if len(sentence) > 20 and len(points) < 5:
#                     # Clean up sentence
#                     if not sentence.endswith('.'):
#                         sentence += '.'
#                     points.append(sentence)
        
#         return points[:5]  # Return top 5 points
    
#     @staticmethod
#     def _format_citations(citations: List[Dict]) -> str:
#         """Format citations professionally."""
        
#         if not citations:
#             return ""
        
#         formatted = "### 📚 Sources Used\n\n"
        
#         for i, citation in enumerate(citations[:5], 1):
#             file_name = citation.get('file', 'Unknown')
#             meeting_type = citation.get('type', '')
#             meeting_date = citation.get('date', 'N/A')
#             score = citation.get('score', 0)
            
#             # Format with confidence score
#             confidence = f"{int(score * 100)}%" if isinstance(score, float) else "N/A"
            
#             formatted += f"**{i}. {file_name}**\n"
#             formatted += f"   - Type: {meeting_type}\n"
#             formatted += f"   - Date: {meeting_date}\n"
#             formatted += f"   - Relevance: {confidence}\n\n"
        
#         return formatted
    
#     @staticmethod
#     def format_comparison_response(
#         entities: List[str],
#         comparisons: Dict,
#         citations: List[Dict]
#     ) -> str:
#         """Format comparison-type answers professionally."""
        
#         formatted = "## 📊 Comparison Analysis\n\n"
        
#         # Create comparison table
#         formatted += "| Aspect | "
#         formatted += " | ".join(entities) + " |\n"
#         formatted += "|--------|" + "|".join(["-----"] * len(entities)) + "|\n"
        
#         # Add comparison rows
#         for aspect, values in comparisons.items():
#             row = f"| **{aspect}** | "
#             row += " | ".join([str(v) for v in values])
#             row += " |\n"
#             formatted += row
        
#         formatted += "\n"
#         formatted += ResponseFormatter._format_citations(citations)
        
#         return formatted
    
#     @staticmethod
#     def format_qa_response(
#         question: str,
#         answer: str,
#         related_questions: List[str] = None
#     ) -> str:
#         """Format Q&A style response."""
        
#         formatted = f"## ❓ Question\n**{question}**\n\n"
#         formatted += f"## ✅ Answer\n{answer}\n\n"
        
#         if related_questions:
#             formatted += "### 🔗 Related Questions\n\n"
#             for i, q in enumerate(related_questions[:3], 1):
#                 formatted += f"{i}. {q}\n"
#             formatted += "\n"
        
#         return formatted
    
#     @staticmethod
#     def format_list_response(
#         title: str,
#         items: List[str],
#         item_type: str = "bullet"
#     ) -> str:
#         """Format list-type responses."""
        
#         formatted = f"## 📋 {title}\n\n"
        
#         if item_type == "bullet":
#             for item in items:
#                 formatted += f"- {item}\n"
#         elif item_type == "numbered":
#             for i, item in enumerate(items, 1):
#                 formatted += f"{i}. {item}\n"
#         elif item_type == "definition":
#             for item in items:
#                 if ':' in item:
#                     term, definition = item.split(':', 1)
#                     formatted += f"**{term.strip()}**: {definition.strip()}\n\n"
        
#         return formatted
    
#     @staticmethod
#     def format_timeline_response(
#         title: str,
#         events: List[Dict]
#     ) -> str:
#         """Format timeline-type responses."""
        
#         formatted = f"## 📅 {title}\n\n"
        
#         for event in events:
#             date = event.get('date', 'N/A')
#             title = event.get('title', 'Event')
#             description = event.get('description', '')
            
#             formatted += f"### {date}: {title}\n"
#             formatted += f"{description}\n\n"
        
#         return formatted
    
#     @staticmethod
#     def format_structured_answer(
#         heading: str,
#         subheadings: Dict[str, str],
#         citations: List[Dict]
#     ) -> str:
#         """
#         Format answer with multiple subheadings.
        
#         Example:
#         {
#             "Overview": "Main content...",
#             "Role": "Detailed role...",
#             "Contributions": "What they contributed..."
#         }
#         """
        
#         formatted = f"## {heading}\n\n"
        
#         for subheading, content in subheadings.items():
#             formatted += f"### {subheading}\n"
#             formatted += f"{content}\n\n"
        
#         formatted += ResponseFormatter._format_citations(citations)
        
#         return formatted
    
#     @staticmethod
#     def add_visual_separators(text: str) -> str:
#         """Add visual separators for better readability."""
        
#         # Add separator after main heading
#         text = text.replace("## ", "## ")
        
#         return text
    
#     @staticmethod
#     def convert_to_html(markdown_text: str) -> str:
#         """Convert markdown to HTML (basic)."""
        
#         html = markdown_text
        
#         # Headers
#         html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
#         html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
#         html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        
#         # Bold
#         html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        
#         # Italic
#         html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        
#         # Lists
#         html = re.sub(r'^- (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
#         html = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', html, flags=re.DOTALL)
        
#         # Line breaks
#         html = html.replace('\n\n', '</p><p>')
#         html = f"<p>{html}</p>"
        
#         return html


# # Test cases
# if __name__ == "__main__":
#     formatter = ResponseFormatter()
    
#     # Test 1: Format answer with structure
#     sample_answer = """
#     Madam Shahnaz Wazir Ali played a significant role in the 15th AC meeting. 
#     She chaired the meeting as the Acting President. 
#     She made suggestions about curriculum improvements. 
#     She approved several important decisions.
#     """
    
#     sample_citations = [
#         {
#             "file": "15th Minutes of the Academic Meeting",
#             "type": "AC",
#             "date": "2015-06-11",
#             "score": 0.92
#         },
#         {
#             "file": "AC Meeting Notes",
#             "type": "AC",
#             "date": "2015-06-11",
#             "score": 0.85
#         }
#     ]
    
#     formatted = formatter.format_answer_with_structure(
#         sample_answer,
#         "What is Madam Shahnaz's role?",
#         sample_citations
#     )
    
#     print("=== FORMATTED RESPONSE ===\n")
#     print(formatted)
    
#     # Test 2: Structured answer
#     structured = formatter.format_structured_answer(
#         "🎯 Madam Shahnaz Wazir Ali's Role in AC 15 Meeting",
#         {
#             "Leadership Role": "She chaired the meeting as Acting President and initiated discussions.",
#             "Key Decisions": "Approved changes to BS-Bio Sciences curriculum and ELM program.",
#             "Contributions": "Made suggestions for curriculum modernization and program improvements."
#         },
#         sample_citations
#     )
    
#     print("\n=== STRUCTURED RESPONSE ===\n")
#     print(structured)


#----------------------------------------------------------------------------------------------


# #!/usr/bin/env python3
# """
# response_formatter.py - Converts answers to beautifully formatted HTML
# """

# import re
# from typing import List, Dict, Any


# class ResponseFormatter:
#     """Formats RAG answers with structure and converts to HTML"""
    
#     def format_answer_with_structure(
#         self,
#         answer: str,
#         query: str = "",
#         citations: List[Dict[str, Any]] = None,
#         confidence_level: str = "high"
#     ) -> str:
#         """
#         Format answer with proper structure
        
#         Args:
#             answer: Raw LLM response
#             query: Original user query
#             citations: List of citations
#             confidence_level: Confidence level (high/medium/low)
        
#         Returns:
#             Formatted markdown-style answer
#         """
#         if not answer:
#             return ""
        
#         formatted = answer.strip()
        
#         # Ensure proper spacing between sections
#         formatted = re.sub(r'\n{3,}', '\n\n', formatted)
        
#         # Ensure numbered/bulleted lists have proper formatting
#         formatted = self._format_lists(formatted)
        
#         # Ensure headings are properly marked
#         formatted = self._format_headings(formatted)
        
#         return formatted
    
#     def _format_headings(self, text: str) -> str:
#         """Ensure headings are properly formatted"""
#         # If line starts with number followed by dot and text, make it a heading
#         text = re.sub(
#             r'^(\d+)\.?\s+([A-Z][^:\n]*?)(?:\s*:)?$',
#             r'## \2',
#             text,
#             flags=re.MULTILINE
#         )
        
#         return text
    
#     def _format_lists(self, text: str) -> str:
#         """Ensure lists are properly formatted"""
#         # Ensure bullet points have proper indentation
#         lines = text.split('\n')
#         formatted_lines = []
        
#         for line in lines:
#             # Check if line is a bullet point
#             if re.match(r'^\s*[-•*]\s+', line):
#                 # Ensure consistent bullet format
#                 line = re.sub(r'^\s*[-•*]\s+', '• ', line)
            
#             formatted_lines.append(line)
        
#         return '\n'.join(formatted_lines)
    
#     def convert_to_html(self, formatted_answer: str) -> str:
#         """
#         Convert formatted answer to HTML
        
#         Args:
#             formatted_answer: Formatted markdown-style answer
        
#         Returns:
#             HTML formatted answer
#         """
#         if not formatted_answer:
#             return ""
        
#         html = formatted_answer
        
#         # Convert ## headings to h2
#         html = re.sub(
#             r'^##\s+(.+?)$',
#             r'<h2>\1</h2>',
#             html,
#             flags=re.MULTILINE
#         )
        
#         # Convert # headings to h1
#         html = re.sub(
#             r'^#\s+([^#].+?)$',
#             r'<h1>\1</h1>',
#             html,
#             flags=re.MULTILINE
#         )
        
#         # Convert **bold** to <strong>
#         html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        
#         # Convert *italic* to <em>
#         html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        
#         # Convert numbered lists: "1. Item" or "1) Item"
#         html = re.sub(
#             r'^(\d+)[\.\)]\s+(.+?)$',
#             r'<li>\2</li>',
#             html,
#             flags=re.MULTILINE
#         )
        
#         # Wrap consecutive <li> tags in <ol>
#         html = re.sub(
#             r'(<li>.*?</li>)',
#             lambda m: '<ol>\n' + m.group(1) + '\n</ol>' if not re.search(r'<ol>', m.group(0)) else m.group(1),
#             html,
#             flags=re.DOTALL
#         )
        
#         # Better approach: find all li groups and wrap them
#         html = self._wrap_list_items(html)
        
#         # Convert bullet points: "• Item"
#         html = re.sub(
#             r'^\s*•\s+(.+?)$',
#             r'<li>\1</li>',
#             html,
#             flags=re.MULTILINE
#         )
        
#         # Wrap consecutive bullet <li> tags in <ul>
#         html = self._wrap_bullet_items(html)
        
#         # Convert line breaks
#         html = re.sub(r'\n\n+', '</p><p>', html)
#         html = f'<p>{html}</p>'
        
#         # Remove empty paragraphs
#         html = re.sub(r'<p>\s*</p>', '', html)
        
#         # Ensure proper spacing for headings
#         html = re.sub(r'</p>(<h[1-6])', r'</p>\n\n\1', html)
#         html = re.sub(r'(</h[1-6]>)<p>', r'\1\n\n<p>', html)
        
#         return html
    
#     def _wrap_list_items(self, html: str) -> str:
#         """Wrap consecutive numbered <li> items in <ol>"""
#         # Find all consecutive <li> lines that look like numbered items
#         lines = html.split('\n')
#         result = []
#         in_list = False
        
#         for line in lines:
#             if '<li>' in line and re.search(r'^\d+', line):
#                 if not in_list:
#                     result.append('<ol>')
#                     in_list = True
#                 result.append(line)
#             else:
#                 if in_list:
#                     result.append('</ol>')
#                     in_list = False
#                 result.append(line)
        
#         if in_list:
#             result.append('</ol>')
        
#         return '\n'.join(result)
    
#     def _wrap_bullet_items(self, html: str) -> str:
#         """Wrap consecutive bullet <li> items in <ul>"""
#         # Find all consecutive <li> lines that came from bullets
#         pattern = r'(<li>.*?</li>)'
        
#         def replace_bullets(match_obj):
#             content = match_obj.group(0)
#             if content.startswith('<li>') and '</li>' in content:
#                 # Check if this is a bullet (not numbered)
#                 inner = content.replace('<li>', '').replace('</li>', '')
#                 if not re.match(r'^\d+\.?', inner):
#                     return content
#             return match_obj.group(0)
        
#         # Simpler approach: wrap all remaining li items
#         html = re.sub(
#             r'(<li>[^<]+</li>\n?)+',
#             lambda m: '<ul>\n' + m.group(0) + '</ul>\n',
#             html
#         )
        
#         return html
    
#     def format_with_citations(
#         self,
#         answer: str,
#         citations: List[Dict[str, Any]]
#     ) -> str:
#         """
#         Format answer with citations section
        
#         Args:
#             answer: Formatted answer
#             citations: List of citations
        
#         Returns:
#             Answer with citations appended
#         """
#         if not citations:
#             return answer
        
#         citations_html = '<hr><h3>📚 Sources</h3><ul>'
        
#         for citation in citations[:5]:  # Show top 5 citations
#             file_name = citation.get('file', 'Unknown')
#             date = citation.get('date', 'Unknown date')
#             score = citation.get('score', 0)
            
#             # Format score as percentage
#             score_pct = int(score * 100) if isinstance(score, float) else score
            
#             citations_html += f'<li><strong>{file_name}</strong> - {date} ({score_pct}% relevance)</li>'
        
#         citations_html += '</ul>'
        
#         return answer + citations_html


# # For backwards compatibility
# def format_response(answer: str, citations: List[Dict] = None) -> str:
#     """Standalone function to format response"""
#     formatter = ResponseFormatter()
#     formatted = formatter.format_answer_with_structure(answer, citations=citations or [])
#     return formatter.convert_to_html(formatted)





#!/usr/bin/env python3
"""
response_formatter.py - Converts answers to beautifully formatted HTML
"""

import re
from typing import List, Dict, Any


class ResponseFormatter:
    """Formats RAG answers with structure and converts to HTML"""

    def format_answer_with_structure(
        self,
        answer: str,
        query: str = "",
        citations: List[Dict[str, Any]] = None,
        confidence_level: str = "high"
    ) -> str:
        """
        Format answer with proper structure.

        Returns a markdown-style string (with #, ##, ###, lists, etc.).
        """
        if not answer:
            return ""

        formatted = answer.strip()

        # Normalize excessive blank lines
        formatted = re.sub(r'\n{3,}', '\n\n', formatted)

        # Normalize bullet list markers
        formatted = self._format_lists(formatted)

        # Optional: promote pure numbered-title lines to headings
        formatted = self._format_headings(formatted)

        return formatted

    def _format_headings(self, text: str) -> str:
        """Optionally convert lines like '1. Title' into '## Title'."""
        text = re.sub(
            r'^(\d+)\.?\s+([A-Z][^:\n]*?)(?:\s*:)?$',
            r'## \2',
            text,
            flags=re.MULTILINE
        )
        return text

    def _format_lists(self, text: str) -> str:
        """Normalize bullet markers to a single style '• '."""
        lines = text.split('\n')
        formatted_lines = []

        for line in lines:
            if re.match(r'^\s*[-•*]\s+', line):
                # Normalize any '-', '*', '•' to '• '
                line = re.sub(r'^\s*[-•*]\s+', '• ', line)
            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def convert_to_html(self, formatted_answer: str) -> str:
        """
        Convert formatted markdown-style answer to HTML.
        Handles headings (#/##/###), bold, italics, ordered/bullet lists, and paragraphs.
        """
        if not formatted_answer:
            return ""

        html = formatted_answer

        # --- Headings ---
        # ### -> h3
        html = re.sub(
            r'^###\s+(.+?)$',
            r'<h3>\1</h3>',
            html,
            flags=re.MULTILINE
        )
        # ## -> h2
        html = re.sub(
            r'^##\s+(.+?)$',
            r'<h2>\1</h2>',
            html,
            flags=re.MULTILINE
        )
        # # (but not ##/###) -> h1
        html = re.sub(
            r'^#\s+([^#].+?)$',
            r'<h1>\1</h1>',
            html,
            flags=re.MULTILINE
        )

        # --- Inline formatting ---
        # Bold: **text**
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        # Italic: *text*
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # --- Ordered lists ---
        # First convert "1. Item" or "1) Item" -> <li>Item</li>
        html = re.sub(
            r'^(\d+)[\.\)]\s+(.+?)$',
            r'<li>\2</li>',
            html,
            flags=re.MULTILINE
        )

        # Now wrap consecutive <li> blocks (that came from numbers) into one <ol>
        # We do this before bullets so both types can coexist.
        html = re.sub(
            r'((?:^<li>.*?</li>\n?)+)',
            lambda m: '<ol>\n' + m.group(1) + '</ol>',
            html,
            flags=re.MULTILINE
        )

        # --- Bullet lists ---
        # Convert "• Item" -> <li>Item</li>
        html = re.sub(
            r'^\s*•\s+(.+?)$',
            r'<li>\1</li>',
            html,
            flags=re.MULTILINE
        )

        # Wrap consecutive remaining <li> groups (bullets) into <ul>
        html = re.sub(
            r'((?:^<li>.*?</li>\n?)+)',
            lambda m: '<ul>\n' + m.group(1) + '</ul>',
            html,
            flags=re.MULTILINE
        )

        # --- Paragraphs ---
        # Convert double newlines into paragraph breaks
        html = re.sub(r'\n\n+', '</p><p>', html)
        html = f'<p>{html}</p>'

        # Remove empty paragraphs
        html = re.sub(r'<p>\s*</p>', '', html)

        # Ensure extra spacing before/after headings
        html = re.sub(r'</p>(<h[1-6])', r'</p>\n\n\1', html)
        html = re.sub(r'(</h[1-6]>)<p>', r'\1\n\n<p>', html)

        return html

    def format_with_citations(
        self,
        answer: str,
        citations: List[Dict[str, Any]]
    ) -> str:
        """Append a simple citations section to an already-HTML answer."""
        if not citations:
            return answer

        citations_html = '<hr><h3>📚 Sources</h3><ul>'

        for citation in citations[:5]:
            file_name = citation.get('file', 'Unknown')
            date = citation.get('date', 'Unknown date')
            score = citation.get('score', 0)
            score_pct = int(score * 100) if isinstance(score, float) else score
            citations_html += (
                f'<li><strong>{file_name}</strong> - {date} '
                f'({score_pct}% relevance)</li>'
            )

        citations_html += '</ul>'
        return answer + citations_html


# For backwards compatibility
def format_response(answer: str, citations: List[Dict] = None) -> str:
    """Standalone function to format response."""
    formatter = ResponseFormatter()
    formatted = formatter.format_answer_with_structure(
        answer,
        citations=citations or []
    )
    return formatter.convert_to_html(formatted)
