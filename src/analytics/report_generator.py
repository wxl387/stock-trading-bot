"""
PDF Report Generator for Portfolio Analytics.
Generates professional performance reports using reportlab.
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
from io import BytesIO
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Report colors
COLORS = {
    'primary': (0.12, 0.47, 0.71),      # Blue
    'success': (0.17, 0.63, 0.17),       # Green
    'danger': (0.84, 0.15, 0.16),        # Red
    'warning': (1.0, 0.50, 0.0),         # Orange
    'dark': (0.2, 0.2, 0.2),             # Dark gray
    'light': (0.95, 0.95, 0.95),         # Light gray
    'white': (1.0, 1.0, 1.0),
}


class ReportGenerator:
    """
    Generate PDF performance reports.

    Creates professional reports with:
    - Executive summary with key metrics
    - Performance charts
    - Monthly returns table
    - Position attribution
    - Trade statistics
    """

    def __init__(self, title: str = "Portfolio Performance Report"):
        """
        Initialize ReportGenerator.

        Args:
            title: Report title.
        """
        self.title = title
        self._reportlab_available = self._check_reportlab()

    def _check_reportlab(self) -> bool:
        """Check if reportlab is available."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate
            return True
        except ImportError:
            logger.warning("reportlab not installed. Install with: pip install reportlab")
            return False

    def generate_report(
        self,
        metrics: Dict,
        benchmark_metrics: Optional[Dict] = None,
        monthly_returns: Optional[pd.DataFrame] = None,
        attribution_data: Optional[Dict] = None,
        trade_analysis: Optional[Dict] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        output_path: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Generate PDF report.

        Args:
            metrics: Core portfolio metrics.
            benchmark_metrics: Benchmark comparison metrics.
            monthly_returns: Monthly returns DataFrame.
            attribution_data: Performance attribution data.
            trade_analysis: Trade statistics.
            start_date: Report period start.
            end_date: Report period end.
            output_path: Optional file path to save report.

        Returns:
            PDF bytes if successful, None otherwise.
        """
        if not self._reportlab_available:
            logger.error("Cannot generate report: reportlab not installed")
            return None

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak, HRFlowable
            )

            # Create buffer or file
            if output_path:
                buffer = output_path
            else:
                buffer = BytesIO()

            # Create document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )

            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                spaceAfter=20,
                textColor=colors.HexColor('#1E3A5F')
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceBefore=15,
                spaceAfter=10,
                textColor=colors.HexColor('#2E5A7F')
            )
            normal_style = styles['Normal']

            # Build content
            content = []

            # Title
            content.append(Paragraph(self.title, title_style))

            # Period
            period_start = start_date.strftime('%Y-%m-%d') if start_date else 'N/A'
            period_end = end_date.strftime('%Y-%m-%d') if end_date else 'N/A'
            content.append(Paragraph(f"Period: {period_start} to {period_end}", normal_style))
            content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))

            content.append(Spacer(1, 20))
            content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CCCCCC')))
            content.append(Spacer(1, 10))

            # Executive Summary
            content.append(Paragraph("Executive Summary", heading_style))
            summary_data = self._build_summary_table(metrics, benchmark_metrics)
            summary_table = Table(summary_data, colWidths=[1.8*inch, 1.2*inch, 1.8*inch, 1.2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5A7F')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
            ]))
            content.append(summary_table)
            content.append(Spacer(1, 20))

            # Risk Metrics
            content.append(Paragraph("Risk-Adjusted Metrics", heading_style))
            risk_data = self._build_risk_table(metrics)
            risk_table = Table(risk_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1*inch, 1.5*inch, 1*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5A7F')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            content.append(risk_table)
            content.append(Spacer(1, 20))

            # Benchmark Comparison
            if benchmark_metrics:
                content.append(Paragraph("Benchmark Comparison (vs SPY)", heading_style))
                bench_data = self._build_benchmark_table(benchmark_metrics)
                bench_table = Table(bench_data, colWidths=[1.5*inch, 1*inch, 1.5*inch, 1*inch, 1.5*inch, 1*inch])
                bench_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5A7F')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                ]))
                content.append(bench_table)
                content.append(Spacer(1, 20))

            # Monthly Returns
            if monthly_returns is not None and not monthly_returns.empty:
                content.append(Paragraph("Monthly Returns (%)", heading_style))
                monthly_data = self._build_monthly_table(monthly_returns)
                col_width = 0.55 * inch
                monthly_table = Table(
                    monthly_data,
                    colWidths=[0.6*inch] + [col_width] * 12
                )
                monthly_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5A7F')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#2E5A7F')),
                    ('TEXTCOLOR', (0, 1), (0, -1), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                ]))
                # Color cells based on positive/negative
                for row_idx, row in enumerate(monthly_data[1:], start=1):
                    for col_idx, cell in enumerate(row[1:], start=1):
                        if cell and cell != '-':
                            try:
                                val = float(cell)
                                if val > 0:
                                    monthly_table.setStyle(TableStyle([
                                        ('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx),
                                         colors.HexColor('#C8E6C9'))
                                    ]))
                                elif val < 0:
                                    monthly_table.setStyle(TableStyle([
                                        ('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx),
                                         colors.HexColor('#FFCDD2'))
                                    ]))
                            except (ValueError, TypeError):
                                pass
                content.append(monthly_table)
                content.append(Spacer(1, 20))

            # Trade Analysis
            if trade_analysis:
                content.append(Paragraph("Trade Analysis", heading_style))
                trade_data = self._build_trade_table(trade_analysis)
                trade_table = Table(trade_data, colWidths=[1.5*inch, 1.2*inch, 1.5*inch, 1.2*inch])
                trade_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5A7F')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                ]))
                content.append(trade_table)
                content.append(Spacer(1, 20))

            # Attribution
            if attribution_data and attribution_data.get('top_contributors'):
                content.append(Paragraph("Top Contributors", heading_style))
                attr_data = self._build_attribution_table(attribution_data['top_contributors'])
                attr_table = Table(attr_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
                attr_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5A7F')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                ]))
                content.append(attr_table)

            # Footer
            content.append(Spacer(1, 30))
            content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CCCCCC')))
            content.append(Spacer(1, 10))
            footer_style = ParagraphStyle(
                'Footer',
                parent=styles['Normal'],
                fontSize=8,
                textColor=colors.HexColor('#666666')
            )
            content.append(Paragraph(
                "This report is generated automatically. Past performance does not guarantee future results.",
                footer_style
            ))

            # Build PDF
            doc.build(content)

            if isinstance(buffer, BytesIO):
                return buffer.getvalue()
            return None

        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return None

    def _build_summary_table(self, metrics: Dict, benchmark: Optional[Dict]) -> List[List[str]]:
        """Build executive summary table data."""
        total_return = metrics.get('total_return', 0)
        ann_return = metrics.get('annualized_return', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)

        vs_spy = benchmark.get('relative_performance', 0) if benchmark else 0

        return [
            ['Metric', 'Value', 'Metric', 'Value'],
            ['Total Return', f"{total_return:.2%}", 'Sharpe Ratio', f"{sharpe:.2f}"],
            ['Ann. Return', f"{ann_return:.2%}", 'Max Drawdown', f"{max_dd:.2%}"],
            ['vs SPY', f"{vs_spy:+.2%}", 'Trading Days', f"{metrics.get('trading_days', 0)}"],
        ]

    def _build_risk_table(self, metrics: Dict) -> List[List[str]]:
        """Build risk metrics table data."""
        return [
            ['Sharpe', 'Value', 'Sortino', 'Value', 'Calmar', 'Value'],
            [
                'Ratio',
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Ratio',
                f"{metrics.get('sortino_ratio', 0):.2f}",
                'Ratio',
                f"{metrics.get('calmar_ratio', 0):.2f}"
            ],
        ]

    def _build_benchmark_table(self, benchmark: Dict) -> List[List[str]]:
        """Build benchmark comparison table data."""
        return [
            ['Alpha', 'Value', 'Beta', 'Value', 'Info Ratio', 'Value'],
            [
                'Annual',
                f"{benchmark.get('alpha', 0):.2%}",
                'Coefficient',
                f"{benchmark.get('beta', 1):.2f}",
                'IR',
                f"{benchmark.get('information_ratio', 0):.2f}"
            ],
        ]

    def _build_monthly_table(self, monthly: pd.DataFrame) -> List[List[str]]:
        """Build monthly returns table data."""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        header = ['Year'] + months
        data = [header]

        for year in monthly.index:
            row = [str(year)]
            for month in monthly.columns:
                val = monthly.loc[year, month]
                if pd.isna(val):
                    row.append('-')
                else:
                    row.append(f"{val*100:.1f}")
            data.append(row)

        return data

    def _build_trade_table(self, trade_analysis: Dict) -> List[List[str]]:
        """Build trade analysis table data."""
        return [
            ['Metric', 'Value', 'Metric', 'Value'],
            [
                'Total Trades',
                f"{trade_analysis.get('total_trades', 0)}",
                'Win Rate',
                f"{trade_analysis.get('win_rate', 0):.1%}"
            ],
            [
                'Avg Win',
                f"${trade_analysis.get('avg_win', 0):,.2f}",
                'Avg Loss',
                f"${trade_analysis.get('avg_loss', 0):,.2f}"
            ],
            [
                'Largest Win',
                f"${trade_analysis.get('largest_win', 0):,.2f}",
                'Largest Loss',
                f"${trade_analysis.get('largest_loss', 0):,.2f}"
            ],
            [
                'Profit Factor',
                f"{trade_analysis.get('profit_factor', 0):.2f}",
                'Win/Loss Ratio',
                f"{trade_analysis.get('avg_win_loss_ratio', 0):.2f}"
            ],
        ]

    def _build_attribution_table(self, contributors: List[Dict]) -> List[List[str]]:
        """Build attribution table data."""
        data = [['Symbol', 'P&L', 'Contribution']]

        for c in contributors:
            pnl = c.get('pnl', 0)
            contrib = c.get('contribution_pct', 0)
            data.append([
                c.get('symbol', ''),
                f"${pnl:,.2f}",
                f"{contrib:.1%}"
            ])

        return data
