import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class ChartUtils:
    def __init__(self):
        """Inisialisasi utilitas grafik dengan skema warna yang ditentukan."""
        self.colors = {
            'primary': '#1f77b4', 'secondary': '#ff7f0e', 'success': '#2ca02c',
            'danger': '#d62728', 'warning': '#ff9800', 'info': '#17a2b8',
            'purple': '#9467bd', 'brown': '#8c564b', 'pink': '#e377c2', 'gray': '#7f7f7f'
        }

    def create_enhanced_charts(self, platform):
        """Membuat grafik analisis teknikal yang ditingkatkan untuk IHSG."""
        if 'IHSG' not in platform.data or platform.data['IHSG'].empty:
            st.warning("⚠️ Data IHSG tidak tersedia untuk membuat grafik.")
            return None

        ihsg_data_full = platform.data['IHSG']
        features = platform.calculate_technical_indicators(ihsg_data_full).add_prefix('ihsg_')
        recent_data = ihsg_data_full.tail(90).copy()
        combined_data = recent_data.join(features)

        if combined_data.empty:
            st.warning("Data gabungan (harga + indikator) kosong.")
            return None
        
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            subplot_titles=(
                'Pergerakan Harga dengan Indikator Trend (SMA & Bollinger Bands)', 
                'Volume Perdagangan Harian', 
                'Relative Strength Index (RSI) - Momentum Overbought/Oversold', 
                'MACD - Konvergensi dan Divergensi Moving Average'
            ),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )

        # Candlestick chart untuk harga
        fig.add_trace(go.Candlestick(
            x=combined_data.index, open=combined_data['Open'], high=combined_data['High'],
            low=combined_data['Low'], close=combined_data['Close'], name='Harga IHSG',
            increasing_line_color=self.colors['success'], decreasing_line_color=self.colors['danger'],
            legendgroup='price', showlegend=True
        ), row=1, col=1)

        # Simple Moving Averages
        for ma_col, color_key, name in [('ihsg_sma_10', 'secondary', 'SMA 10 Hari'), ('ihsg_sma_20', 'purple', 'SMA 20 Hari')]:
            if ma_col in combined_data.columns:
                fig.add_trace(go.Scatter(
                    x=combined_data.index, y=combined_data[ma_col], mode='lines',
                    name=name, line=dict(color=self.colors[color_key], width=1.8), opacity=0.8,
                    legendgroup='sma', showlegend=True
                ), row=1, col=1)
        
        # Bollinger Bands
        if all(col in combined_data.columns for col in ['ihsg_bb_upper', 'ihsg_bb_lower']):
            fig.add_trace(go.Scatter(
                x=combined_data.index, y=combined_data['ihsg_bb_upper'], mode='lines',
                name='Bollinger Band Atas', line=dict(color=self.colors['gray'], width=1.2, dash='dot'), 
                opacity=0.7, legendgroup='bb', showlegend=True
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=combined_data.index, y=combined_data['ihsg_bb_lower'], mode='lines',
                name='Bollinger Band Bawah', line=dict(color=self.colors['gray'], width=1.2, dash='dot'),
                opacity=0.7, fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                legendgroup='bb', showlegend=True
            ), row=1, col=1)

        # Volume dengan warna berdasarkan bullish/bearish
        is_bullish = combined_data['Close'] >= combined_data['Open']
        volume_colors = np.where(is_bullish, self.colors['success'], self.colors['danger'])
        fig.add_trace(go.Bar(
            x=combined_data.index, y=combined_data['ihsg_volume'], name='Volume Transaksi',
            marker_color=volume_colors, opacity=0.7, legendgroup='volume', showlegend=True
        ), row=2, col=1)

        # RSI dengan level overbought/oversold
        if 'ihsg_rsi_14' in combined_data.columns:
            fig.add_trace(go.Scatter(
                x=combined_data.index, y=combined_data['ihsg_rsi_14'], mode='lines',
                name='RSI (14 Periode)', line=dict(color=self.colors['info'], width=2.2),
                legendgroup='rsi', showlegend=True
            ), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.6, row=3, col=1, 
                         annotation_text="Overbought (70)", annotation_position="top right")
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.6, row=3, col=1, 
                         annotation_text="Oversold (30)", annotation_position="bottom right")

        # MACD dengan histogram
        if all(col in combined_data.columns for col in ['ihsg_macd', 'ihsg_macd_signal', 'ihsg_macd_hist']):
            fig.add_trace(go.Scatter(
                x=combined_data.index, y=combined_data['ihsg_macd'], mode='lines',
                name='MACD Line', line=dict(color=self.colors['primary'], width=2),
                legendgroup='macd', showlegend=True
            ), row=4, col=1)
            fig.add_trace(go.Scatter(
                x=combined_data.index, y=combined_data['ihsg_macd_signal'], mode='lines',
                name='Signal Line', line=dict(color=self.colors['secondary'], width=2),
                legendgroup='macd', showlegend=True
            ), row=4, col=1)
            histogram_colors = np.where(combined_data['ihsg_macd_hist'] >= 0, self.colors['success'], self.colors['danger'])
            fig.add_trace(go.Bar(
                x=combined_data.index, y=combined_data['ihsg_macd_hist'],
                name='MACD Histogram', marker_color=histogram_colors, opacity=0.7,
                legendgroup='macd', showlegend=True
            ), row=4, col=1)

        fig.update_layout(
            title={
                'text': 'Dashboard Analisis Teknikal IHSG - 90 Hari Terakhir', 
                'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}
            },
            template='plotly_white', height=850, 
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                font=dict(size=10)
            ),
            margin=dict(r=120) 
        )
        
        for i, annotation in enumerate(fig['layout']['annotations']):
            annotation.update(
                x=0.5,
                xanchor='center',
                font=dict(size=12, color='#2c3e50'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.1)',
                borderwidth=1,
                borderpad=4
            )
            
        return fig

    def create_backtest_chart(self, platform):
        if not hasattr(platform, 'backtest_results') or not platform.backtest_results:
            return None
        y_test = platform.backtest_results.get('y_test')
        predictions_dict = platform.backtest_results.get('predictions', {})
        if y_test is None or y_test.empty or not predictions_dict:
            return None
        
        fig = go.Figure()
        
        # Harga aktual
        fig.add_trace(go.Scatter(
            x=y_test.index, y=y_test, mode='lines',
            name='Harga Aktual (Data Testing)', line=dict(color=self.colors['primary'], width=3),
            legendgroup='actual'
        ))
        
        # Prediksi model
        model_colors = {'XGBoost': self.colors['danger'], 'LightGBM': self.colors['success']}
        for model_name, pred_series in predictions_dict.items():
            if pred_series is not None and not pred_series.empty:
                fig.add_trace(go.Scatter(
                    x=pred_series.index, y=pred_series, mode='lines',
                    name=f'Prediksi Model {model_name}',
                    line=dict(color=model_colors.get(model_name, self.colors['gray']), width=2.5, dash='dash'),
                    legendgroup='prediction'
                ))
        
        fig.update_layout(
            title={
                'text': 'Evaluasi Performa Model - Perbandingan Harga Aktual vs Prediksi', 
                'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}
            },
            xaxis_title='Tanggal', yaxis_title='Harga IHSG (IDR)',
            template='plotly_white', 
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="center", 
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            ),
            hovermode='x unified', height=520
        )
        return fig

    def create_forecast_chart(self, platform, prediction_results):
        if 'IHSG' not in platform.data or platform.data['IHSG'].empty:
            return None
        ihsg_data = platform.data['IHSG']
        recent_data = ihsg_data['Close'].tail(90).copy()
        if recent_data.empty:
            return None
        
        fig = go.Figure()
        
        # Harga historis
        fig.add_trace(go.Scatter(
            x=recent_data.index, y=recent_data, mode='lines',
            name='Harga Historis (90 Hari)', line=dict(color=self.colors['primary'], width=2.5),
            legendgroup='historical'
        ))
        
        if prediction_results and 'forecast_path' in prediction_results:
            forecast_path = prediction_results['forecast_path']
            forecast_dates = [item['date'] for item in forecast_path]
            forecast_prices = [item['price'] for item in forecast_path]
            full_forecast_dates = [recent_data.index[-1]] + forecast_dates
            full_forecast_prices = [recent_data.iloc[-1]] + forecast_prices
            
            # Hitung confidence interval berdasarkan volatilitas historis dan tingkat kepercayaan
            confidence_level = prediction_results.get('confidence', 80) / 100  # Convert percentage to decimal
            
            # Hitung volatilitas historis (standar deviasi returns)
            returns = recent_data.pct_change().dropna()
            historical_volatility = returns.std()
            
            # Hitung interval kepercayaan yang melebar seiring waktu
            confidence_intervals = []
            for i, (date, price) in enumerate(zip(forecast_dates, forecast_prices)):
                # Volatilitas meningkat dengan akar kuadrat dari waktu
                time_adjusted_volatility = historical_volatility * np.sqrt(i + 1)
                
                # Z-score untuk confidence interval (95% = 1.96, 90% = 1.645, 80% = 1.28)
                z_score = {0.95: 1.96, 0.90: 1.645, 0.80: 1.28, 0.68: 1.0}.get(
                    round(confidence_level, 2), 1.96
                )
                
                # Hitung upper dan lower bound
                price_std = price * time_adjusted_volatility
                upper_bound = price + (z_score * price_std)
                lower_bound = price - (z_score * price_std)
                
                confidence_intervals.append({
                    'date': date,
                    'price': price,
                    'upper': upper_bound,
                    'lower': lower_bound
                })
            
            # Tambahkan titik awal untuk confidence interval
            initial_price = recent_data.iloc[-1]
            full_ci_dates = [recent_data.index[-1]] + [item['date'] for item in confidence_intervals]
            full_upper_bounds = [initial_price] + [item['upper'] for item in confidence_intervals]
            full_lower_bounds = [initial_price] + [item['lower'] for item in confidence_intervals]
            
            # Plot confidence interval sebagai area
            fig.add_trace(go.Scatter(
                x=full_ci_dates, y=full_upper_bounds, mode='lines',
                name=f'Batas Atas ({int(confidence_level*100)}% CI)',
                line=dict(color=self.colors['gray'], width=1, dash='dash'),
                legendgroup='confidence',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=full_ci_dates, y=full_lower_bounds, mode='lines',
                name=f'Selang Kepercayaan {int(confidence_level*100)}%',
                line=dict(color=self.colors['gray'], width=1, dash='dash'),
                fill='tonexty',
                fillcolor=f'rgba(128,128,128,0.2)',
                legendgroup='confidence'
            ))
            
            # Jalur peramalan (plot di atas confidence interval)
            fig.add_trace(go.Scatter(
                x=full_forecast_dates, y=full_forecast_prices, mode='lines+markers',
                name=f'Peramalan {platform.prediction_horizon} Hari',
                line=dict(color=self.colors['info'], width=2.5, dash='dot'),
                marker=dict(size=6, color=self.colors['info']),
                legendgroup='forecast'
            ))
            
            # Titik prediksi akhir
            final_pred_date = forecast_dates[-1]
            final_pred_price = forecast_prices[-1]
            direction = prediction_results['direction']
            marker_color = self.colors['success'] if direction == 'UP' else self.colors['danger']
            marker_symbol = 'arrow-up' if direction == 'UP' else 'arrow-down'
            
            fig.add_trace(go.Scatter(
                x=[final_pred_date], y=[final_pred_price], mode='markers',
                name=f'Target Akhir ({direction})',
                marker=dict(size=16, color=marker_color, symbol=marker_symbol, 
                        line=dict(width=2, color='white')),
                legendgroup='target'
            ))
            
            # Tambahkan annotation untuk confidence interval
            fig.add_annotation(
                text=f"Selang Kepercayaan {int(confidence_level*100)}%<br>menunjukkan range kemungkinan<br>pergerakan harga",
                xref="paper", yref="paper",
                x=0.02, y=0.98, xanchor="left", yanchor="top",
                showarrow=False,
                font=dict(size=10, color="gray"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            )
        
        fig.update_layout(
            title={
                'text': f'Peramalan Harga IHSG dengan Selang Kepercayaan - Horizon {platform.prediction_horizon} Hari ke Depan', 
                'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}
            },
            xaxis_title='Tanggal', yaxis_title='Harga IHSG (IDR)',
            template='plotly_white', height=520,
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="center", 
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1
            ),
            hovermode='x unified'
        )
        return fig

    def create_correlation_chart(self, correlations, prediction_horizon):
        if not correlations:
            return None
        top_correlations = dict(list(correlations.items())[:10])
        assets = list(top_correlations.keys())[::-1]
        values = list(top_correlations.values())[::-1]
        colors = [self.colors['success'] if v > 0 else self.colors['danger'] for v in values]
        
        fig = go.Figure(go.Bar(
            x=values, y=assets, orientation='h',
            marker=dict(color=colors, opacity=0.8),
            text=[f'{v:.3f}' for v in values],
            textposition='outside',
            textfont=dict(size=11, color='black'),
            name='Korelasi dengan IHSG'
        ))
        
        # Tambahkan garis referensi
        fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", line_width=1, opacity=0.3)
        fig.add_vline(x=-0.5, line_dash="dash", line_color="gray", line_width=1, opacity=0.3)
        
        fig.update_layout(
            title={
                'text': f'Analisis Korelasi Pasar - Top 10 Aset vs IHSG (Horizon {prediction_horizon} Hari)', 
                'x': 0.5, 'xanchor': 'center', 'font': {'size': 16}
            },
            xaxis_title='Koefisien Korelasi (-1 hingga +1)', 
            yaxis_title='Instrumen Keuangan',
            template='plotly_white', 
            height=max(450, len(assets) * 45),
            xaxis=dict(range=[-1.1, 1.1], gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            showlegend=False,
            margin=dict(l=120, r=60, t=80, b=120) 
        )
        
        fig.add_annotation(
            text="Korelasi Positif: Bergerak searah dengan IHSG<br>Korelasi Negatif: Bergerak berlawanan dengan IHSG",
            xref="paper", yref="paper",
            x=0.5,           
            y=-0.25,         
            xanchor='center',
            yanchor='top',  
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1
        )
        
        return fig