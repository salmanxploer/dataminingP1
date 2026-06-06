"""
Reporting Module
KDD PHASE 7: Visualization & Reporting

Generate comprehensive reports for KDD analysis:
- Executive summary
- Detailed findings
- Statistical summaries
- Visualizations integration
- PDF/HTML export
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate comprehensive KDD analysis reports
    
    Report Sections:
    - Executive Summary
    - Data Overview
    - Preprocessing Results
    - Feature Engineering Summary
    - Clustering Results
    - Trend Analysis
    - Forecasts
    - Recommendations
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory for report output
        """
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / 'reports')
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_content = []
        logger.info(f"Report generator initialized. Output: {self.output_dir}")

    def add_section(self, title: str, content: str):
        """
        Add section to report
        
        Args:
            title: Section title
            content: Section content
        """
        self.report_content.append({
            'type': 'section',
            'title': title,
            'content': content
        })
        logger.info(f"Section added: {title}")

    def add_table(self, title: str, data: Dict):
        """
        Add table to report
        
        Args:
            title: Table title
            data: Table data (dict or list of dicts)
        """
        self.report_content.append({
            'type': 'table',
            'title': title,
            'data': data
        })
        logger.info(f"Table added: {title}")

    def add_metrics(self, metrics: Dict):
        """
        Add metrics block to report
        
        Args:
            metrics: Dictionary of metrics
        """
        self.report_content.append({
            'type': 'metrics',
            'data': metrics
        })
        logger.info(f"Metrics added: {len(metrics)} items")

    def add_figure(self, figure_path: str, caption: str = ""):
        """
        Add figure to report
        
        Args:
            figure_path: Path to figure file
            caption: Figure caption
        """
        self.report_content.append({
            'type': 'figure',
            'path': figure_path,
            'caption': caption
        })
        logger.info(f"Figure added: {figure_path}")

    def generate_text_report(self) -> str:
        """
        Generate plain text report
        
        Returns:
            Report text content
        """
        report_text = self._generate_header()
        
        for item in self.report_content:
            if item['type'] == 'section':
                report_text += self._format_section(item['title'], item['content'])
            elif item['type'] == 'table':
                report_text += self._format_table(item['title'], item['data'])
            elif item['type'] == 'metrics':
                report_text += self._format_metrics(item['data'])
            elif item['type'] == 'figure':
                report_text += self._format_figure(item['path'], item.get('caption', ''))
        
        report_text += self._generate_footer()
        return report_text

    def _generate_header(self) -> str:
        """Generate report header"""
        header = "=" * 80 + "\n"
        header += "KDD ANALYSIS REPORT\n"
        header += "Knowledge Discovery in Databases\n"
        header += "=" * 80 + "\n\n"
        header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "\n" + "=" * 80 + "\n\n"
        return header

    def _generate_footer(self) -> str:
        """Generate report footer"""
        footer = "\n" + "=" * 80 + "\n"
        footer += "END OF REPORT\n"
        footer += "=" * 80 + "\n"
        return footer

    def _format_section(self, title: str, content: str) -> str:
        """Format section content"""
        formatted = f"\n{title}\n"
        formatted += "-" * len(title) + "\n"
        formatted += f"{content}\n"
        return formatted

    def _format_table(self, title: str, data: Dict) -> str:
        """Format table content"""
        formatted = f"\n{title}\n"
        formatted += "-" * len(title) + "\n"
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, float):
                    formatted += f"  {key}: {value:.4f}\n"
                else:
                    formatted += f"  {key}: {value}\n"
        elif isinstance(data, list):
            for item in data:
                formatted += f"  {item}\n"
        
        formatted += "\n"
        return formatted

    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics block"""
        formatted = "\nKEY METRICS\n"
        formatted += "-" * 50 + "\n"
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                formatted += f"  {metric_name}: {metric_value:.4f}\n"
            elif isinstance(metric_value, dict):
                formatted += f"  {metric_name}:\n"
                for sub_key, sub_value in metric_value.items():
                    if isinstance(sub_value, float):
                        formatted += f"    {sub_key}: {sub_value:.4f}\n"
                    else:
                        formatted += f"    {sub_key}: {sub_value}\n"
            else:
                formatted += f"  {metric_name}: {metric_value}\n"
        
        formatted += "\n"
        return formatted

    def _format_figure(self, figure_path: str, caption: str) -> str:
        """Format figure reference"""
        formatted = f"\n[FIGURE: {figure_path}]\n"
        if caption:
            formatted += f"Caption: {caption}\n"
        formatted += "\n"
        return formatted

    def save_text_report(self, filename: str = None) -> str:
        """
        Save report as text file
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved report
        """
        if filename is None:
            filename = f"kdd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report_path = self.output_dir / filename
        report_text = self.generate_text_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Text report saved: {report_path}")
        return str(report_path)

    def generate_html_report(self) -> str:
        """
        Generate HTML report
        
        Returns:
            HTML content
        """
        html = "<!DOCTYPE html>\n"
        html += "<html>\n"
        html += "<head>\n"
        html += "  <title>KDD Analysis Report</title>\n"
        html += "  <style>\n"
        html += self._get_html_style()
        html += "  </style>\n"
        html += "</head>\n"
        html += "<body>\n"
        
        # Header
        html += "  <div class='header'>\n"
        html += "    <h1>KDD Analysis Report</h1>\n"
        html += f"    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n"
        html += "  </div>\n"
        
        # Content
        for item in self.report_content:
            if item['type'] == 'section':
                html += f"  <div class='section'>\n"
                html += f"    <h2>{item['title']}</h2>\n"
                html += f"    <p>{item['content']}</p>\n"
                html += f"  </div>\n"
            elif item['type'] == 'table':
                html += self._format_html_table(item['title'], item['data'])
            elif item['type'] == 'metrics':
                html += self._format_html_metrics(item['data'])
            elif item['type'] == 'figure':
                html += f"  <div class='figure'>\n"
                html += f"    <img src='{item['path']}' alt='Figure'>\n"
                if item.get('caption'):
                    html += f"    <p class='caption'>{item['caption']}</p>\n"
                html += f"  </div>\n"
        
        html += "</body>\n"
        html += "</html>\n"
        
        return html

    def _format_html_table(self, title: str, data: Dict) -> str:
        """Format HTML table"""
        html = f"  <div class='table'>\n"
        html += f"    <h3>{title}</h3>\n"
        html += f"    <table>\n"
        
        if isinstance(data, dict):
            for key, value in data.items():
                html += f"      <tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html += f"    </table>\n"
        html += f"  </div>\n"
        return html

    def _format_html_metrics(self, metrics: Dict) -> str:
        """Format HTML metrics"""
        html = f"  <div class='metrics'>\n"
        html += f"    <h3>Key Metrics</h3>\n"
        html += f"    <table>\n"
        
        for metric_name, metric_value in metrics.items():
            html += f"      <tr><td>{metric_name}</td><td>{metric_value}</td></tr>\n"
        
        html += f"    </table>\n"
        html += f"  </div>\n"
        return html

    def _get_html_style(self) -> str:
        """Get HTML CSS styles"""
        style = """
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .header h1 {
            margin: 0;
        }
        .section {
            background-color: white;
            padding: 15px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }
        .section h2 {
            color: #2c3e50;
            margin-top: 0;
        }
        .table, .metrics {
            background-color: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        table tr:hover {
            background-color: #f5f5f5;
        }
        .figure {
            background-color: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            text-align: center;
        }
        .figure img {
            max-width: 100%;
            height: auto;
        }
        .caption {
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }
        """
        return style

    def save_html_report(self, filename: str = None) -> str:
        """
        Save report as HTML file
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved report
        """
        if filename is None:
            filename = f"kdd_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        report_path = self.output_dir / filename
        report_html = self.generate_html_report()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"HTML report saved: {report_path}")
        return str(report_path)

    def create_executive_summary(self, data_stats: Dict, clustering_metrics: Dict,
                                trend_insights: List[str]) -> str:
        """
        Create executive summary section
        
        Args:
            data_stats: Data statistics
            clustering_metrics: Clustering evaluation metrics
            trend_insights: List of trend insights
            
        Returns:
            Executive summary text
        """
        summary = "EXECUTIVE SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        
        summary += "Data Overview:\n"
        for key, value in data_stats.items():
            summary += f"  • {key}: {value}\n"
        
        summary += "\nClustering Performance:\n"
        for key, value in clustering_metrics.items():
            if isinstance(value, float):
                summary += f"  • {key}: {value:.4f}\n"
            else:
                summary += f"  • {key}: {value}\n"
        
        summary += "\nKey Insights:\n"
        for i, insight in enumerate(trend_insights, 1):
            summary += f"  {i}. {insight}\n"
        
        return summary

    def clear_report(self):
        """Clear current report content"""
        self.report_content = []
        logger.info("Report content cleared")
