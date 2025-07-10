import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
import traceback

warnings.filterwarnings('ignore')


class IHSGIntelligencePlatform:
    def __init__(self):
        """Inisialisasi platform dengan aset, model, dan parameter yang ditentukan."""
        self.ASSETS = {
            '^JKSE': 'IHSG',
            'USDIDR=X': 'USD_IDR',
            '^N225': 'Nikkei',
            '^HSI': 'HangSeng',
            '^GSPC': 'SP500',
            'GC=F': 'Gold',
            'CL=F': 'Oil',
            'BTC-USD': 'Bitcoin',
            '^VIX': 'VIX'
        }
        self.data = {}
        self.models = {}
        self.scaler = StandardScaler()
        self.prediction_horizon = 1
        self.backtest_results = {}
        self.X_latest = None
        self.X_columns = []
        self.are_models_trained = False
        self.model_performance = {}

    def fetch_data(self, period='5y'):
        """Mengunduh data pasar untuk semua aset dengan visualisasi progres."""
        st.info("🔄 Memulai pengunduhan data pasar...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        self.data = {}
        fetched_count = 0

        for i, (symbol, name) in enumerate(self.ASSETS.items()):
            try:
                progress = (i + 1) / len(self.ASSETS)
                progress_bar.progress(progress)
                status_text.text(f"Mengambil data {name} ({i + 1}/{len(self.ASSETS)})...")
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, auto_adjust=True, end=datetime.now())

                if not data.empty and len(data) > 100:
                    data = data.dropna()
                    self.data[name] = data
                    fetched_count += 1
                else:
                    st.warning(f"⚠️ {name}: Data tidak cukup ({len(data) if not data.empty else 0} baris) atau kosong.")
            except Exception as e:
                st.error(f"❌ Gagal mengambil data {name}: {str(e)}")

        progress_bar.empty()
        status_text.empty()
        if fetched_count > 0:
            st.success(f"✅ Berhasil mengunduh data untuk {fetched_count} aset!")
            return True
        else:
            st.error("❌ Gagal memuat data untuk semua aset.")
            return False

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def calculate_bollinger_bands(self, prices, window=20, num_std_dev=2):
        middle_band = prices.rolling(window=window).mean()
        std_dev = prices.rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std_dev)
        lower_band = middle_band - (std_dev * num_std_dev)
        return middle_band, upper_band, lower_band

    def calculate_atr(self, high, low, close, window=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1/window, adjust=False).mean()
        return atr

    def calculate_market_correlations(self):
        """
        Menghitung korelasi pasar.
        """
        st.info(f"🔍 Menghitung korelasi pasar untuk horizon {self.prediction_horizon} hari...")
        horizon = self.prediction_horizon

        close_prices = {name: data['Close'] for name, data in self.data.items() if not data.empty}
        
        if 'IHSG' not in close_prices:
            st.warning("⚠️ Data IHSG tidak tersedia. Korelasi tidak dapat dihitung.")
            self.correlations = {}
            return self.correlations

        combined_df = pd.DataFrame(close_prices)
        combined_df.ffill(inplace=True)
        combined_df.bfill(inplace=True)

        original_ihsg_dates = self.data['IHSG'].index
        trading_day_df = combined_df.loc[combined_df.index.isin(original_ihsg_dates)]
        returns_df = trading_day_df.pct_change(periods=horizon)

        if len(returns_df) < 30:
            st.warning(f"⚠️ Data tidak cukup untuk menghitung korelasi yang valid (hanya {len(returns_df)} baris).")
            self.correlations = {}
            return self.correlations
            
        correlation_matrix = returns_df.corr()

        if 'IHSG' in correlation_matrix:
            ihsg_correlations = correlation_matrix['IHSG'].drop('IHSG')
            
            sorted_correlations = ihsg_correlations.abs().sort_values(ascending=False)
            self.correlations = correlation_matrix['IHSG'][sorted_correlations.index].to_dict()
        else:
            self.correlations = {}

        if self.correlations:
            st.success(f"✅ Korelasi pasar untuk {len(self.correlations)} aset berhasil dihitung.")
        else:
            st.warning("⚠️ Gagal menghitung korelasi aset.")

        return self.correlations

    def calculate_technical_indicators(self, data):
        if data.empty:
            return pd.DataFrame()
        try:
            indicators = pd.DataFrame(index=data.index)
            close_prices = data['Close']
            high_prices = data['High']
            low_prices = data['Low']
            volume = data['Volume']
            for lag in [1, 2, 3, 5, 7]:
                indicators[f'returns_{lag}'] = close_prices.pct_change(lag)
            for window in [5, 10, 20]:
                indicators[f'volatility_{window}'] = indicators['returns_1'].rolling(window).std()
            indicators['atr_14'] = self.calculate_atr(high_prices, low_prices, close_prices)
            for period in [5, 10, 15]:
                 indicators[f'momentum_{period}'] = close_prices / close_prices.shift(period) - 1
                 indicators[f'roc_{period}'] = close_prices.pct_change(period)
            for window in [10, 20, 50]:
                sma = close_prices.rolling(window).mean()
                indicators[f'sma_{window}'] = sma
                indicators[f'dist_from_sma_{window}'] = (close_prices - sma) / sma
            indicators['rsi_14'] = self.calculate_rsi(close_prices, window=14)
            macd, macd_signal, macd_hist = self.calculate_macd(close_prices)
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist
            bb_mid, bb_upper, bb_lower = self.calculate_bollinger_bands(close_prices)
            indicators['bb_upper'] = bb_upper
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = (bb_upper - bb_lower) / bb_mid
            indicators['volume'] = volume
            indicators['volume_change'] = volume.pct_change()
            indicators['volume_sma_20'] = volume.rolling(20).mean()
            indicators['rsi_x_vol_change'] = indicators['rsi_14'] * indicators['volume_change']
            return indicators.fillna(method='ffill').fillna(0)
        except Exception as e:
            st.warning(f"⚠️ Gagal menghitung indikator teknikal: {str(e)}")
            return pd.DataFrame()

    def create_features(self):
        st.info("🔧 Membangun fitur yang disempurnakan untuk model...")
        if 'IHSG' not in self.data or self.data['IHSG'].empty:
            st.error("Data IHSG tidak tersedia untuk pembuatan fitur.")
            return None, None
        full_date_index = pd.date_range(start=self.data['IHSG'].index.min(), end=self.data['IHSG'].index.max(), freq='B')
        ihsg_temp_df = self.data['IHSG'].reindex(full_date_index).ffill()
        features = self.calculate_technical_indicators(ihsg_temp_df)
        features = features.add_prefix('ihsg_')
        features['time_dayofweek'] = features.index.dayofweek
        features['time_month'] = features.index.month
        features['time_weekofyear'] = features.index.isocalendar().week.astype(int)
        for asset_name, asset_df in self.data.items():
            if asset_name == 'IHSG' or asset_df.empty:
                continue
            asset_aligned = asset_df['Close'].reindex(full_date_index).ffill()
            if asset_aligned.notna().sum() > len(full_date_index) * 0.5:
                for lag in [1, 3, 5, 7]:
                    features[f'{asset_name.lower()}_return_{lag}'] = asset_aligned.pct_change(lag)
        
        # Untuk rolling forecast, kita selalu prediksi 1 hari ke depan
        features = features.shift(1) 
        target = self.data['IHSG']['Close'] 
        
        combined_df = pd.concat([features, target.rename('target')], axis=1)
        combined_df = combined_df.reindex(self.data['IHSG'].index)
        combined_df.dropna(inplace=True)
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
        
        X = combined_df.drop('target', axis=1)
        y = combined_df['target']

        if X.empty or y.empty:
            st.error("❌ Tidak ada fitur valid yang bisa dibuat setelah pembersihan.")
            return None, None
        
        self.X_columns = X.columns.tolist()
        st.success(f"✅ Berhasil membuat {len(X)} sampel dengan {len(X.columns)} fitur untuk peramalan 1-langkah.")
        return X, y

    def train_models(self):
        st.info("🤖 Memulai pelatihan model...")
        X, y = self.create_features()
        if X is None or len(X) < 100:
            st.error("❌ Data tidak cukup untuk melatih model.")
            self.are_models_trained = False
            return None
            
        self.X_latest = X.copy()
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        st.info(f"📊 Set Pelatihan: {len(X_train)} sampel, Set Uji: {len(X_test)} sampel")
        
        self.model_performance = {}
        self.models = {}
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            model_configs = {
                'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1, subsample=0.8, colsample_bytree=0.8),
                'LightGBM': lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1, n_jobs=-1, num_leaves=31)
            }
            
            for model_name, model in model_configs.items():
                st.text(f"Melatih {model_name}...")
                model.fit(X_train_scaled, y_train)
                self.models[model_name] = model
                
            self.backtest_results['y_test'] = y_test
            self.backtest_results['predictions'] = {}
            for model_name, model in self.models.items():
                pred = model.predict(X_test_scaled)
                self.backtest_results['predictions'][model_name] = pd.Series(pred, index=y_test.index)
                
                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)
                
                actual_direction = (y_test > y_test.shift(1)).astype(int)
                pred_direction = (pd.Series(pred, index=y_test.index) > y_test.shift(1)).astype(int)
                valid_indices = actual_direction.notna() & pred_direction.notna()
                directional_accuracy = np.mean(actual_direction[valid_indices] == pred_direction[valid_indices]) * 100
                
                self.model_performance[model_name] = {'Test_MAE': mae, 'Test_RMSE': rmse, 'Test_R2': r2, 'Directional_Accuracy': directional_accuracy}
                
            st.success("✅ Model berhasil dilatih!")
            self.are_models_trained = True
            return self.model_performance
        except Exception as e:
            st.error(f"❌ Pelatihan model gagal: {str(e)}")
            self.are_models_trained = False
            return None

    def _predict_simple_sma(self, price_series: pd.Series, window: int = 5) -> float:
            """Model peramalan sederhana untuk aset eksternal menggunakan SMA."""
            return price_series.tail(window).mean()

    def make_prediction(self):
        """
        Membuat prediksi harga IHSG secara rekursif (rolling forecast).
        Versi ini menerapkan pendekatan bertingkat: meramal aset eksternal di setiap langkah.
        """
        if not self.are_models_trained or not self.models:
            st.error("Model belum dilatih! Silakan latih model terlebih dahulu.")
            return None

        st.info(f"🔮 Memulai peramalan bertingkat untuk {self.prediction_horizon} hari ke depan...")

        try:
            temp_histories = {name: data.copy() for name, data in self.data.items()}
            
            future_predictions = []
            final_individual_predictions = {}
            
            for day in range(self.prediction_horizon):
                last_date = temp_histories['IHSG'].index[-1]
                next_date = last_date + pd.tseries.offsets.BusinessDay(1)
                ihsg_features = self.calculate_technical_indicators(temp_histories['IHSG']).add_prefix('ihsg_')
                external_features = {}
                for asset_name, asset_df in temp_histories.items():
                    if asset_name == 'IHSG':
                        continue
                    # Buat fitur return untuk aset eksternal
                    for lag in [1, 3, 5, 7]:
                        feature_name = f'{asset_name.lower()}_return_{lag}'
                        if feature_name in self.X_columns:
                             external_features[feature_name] = asset_df['Close'].pct_change(lag).iloc[-1]
                
                # Gabungkan semua fitur
                latest_features = pd.concat([
                    ihsg_features.iloc[-1:], 
                    pd.DataFrame(external_features, index=[last_date])
                ], axis=1)
                
                # Tambahkan fitur waktu
                latest_features['time_dayofweek'] = next_date.dayofweek
                latest_features['time_month'] = next_date.month
                latest_features['time_weekofyear'] = next_date.isocalendar().week
                
                # Pastikan urutan kolom sesuai dengan saat training
                latest_features = latest_features.reindex(columns=self.X_columns).fillna(method='ffill')

                features_scaled = self.scaler.transform(latest_features)
                predictions_raw = {name: model.predict(features_scaled)[0] for name, model in self.models.items()}
                next_ihsg_price = np.mean(list(predictions_raw.values()))
                
                future_predictions.append({'date': next_date, 'price': next_ihsg_price})

                if day == self.prediction_horizon - 1:
                    final_individual_predictions = predictions_raw

                new_ihsg_row = pd.DataFrame(
                    {'Open': next_ihsg_price, 'High': next_ihsg_price, 'Low': next_ihsg_price, 'Close': next_ihsg_price, 'Volume': temp_histories['IHSG']['Volume'].iloc[-1]},
                    index=[next_date]
                )
                temp_histories['IHSG'] = pd.concat([temp_histories['IHSG'], new_ihsg_row])
                for asset_name, asset_df in temp_histories.items():
                    if asset_name == 'IHSG':
                        continue
                    
                    next_external_price = self._predict_simple_sma(asset_df['Close'])
                    new_external_row = pd.DataFrame(
                        {'Open': next_external_price, 'High': next_external_price, 'Low': next_external_price, 'Close': next_external_price, 'Volume': asset_df['Volume'].iloc[-1]},
                        index=[next_date]
                    )
                    temp_histories[asset_name] = pd.concat([asset_df, new_external_row])

            current_price = self.data['IHSG']['Close'].iloc[-1]
            final_prediction = future_predictions[-1]
            price_change = final_prediction['price'] - current_price
            predicted_return = price_change / current_price
            std_dev = np.std(list(final_individual_predictions.values()))
            normalized_std_dev = std_dev / current_price
            confidence = max(50, min(99, 100 - (normalized_std_dev * 250)))

            results = {
                'prediction_horizon': self.prediction_horizon, 'current_price': current_price,
                'predicted_price': final_prediction['price'], 'predicted_return': predicted_return,
                'price_change': price_change, 'confidence': confidence, 'direction': 'UP' if price_change > 0 else 'DOWN',
                'magnitude': abs(predicted_return) * 100,
                'individual_predictions': {k: float(v) for k, v in final_individual_predictions.items()},
                'model_performance': self.model_performance, 'forecast_date': final_prediction['date'].strftime('%Y-%m-%d'),
                'forecast_path': future_predictions
            }

            st.success("✅ Peramalan bertingkat berhasil dibuat!")
            return results

        except Exception as e:
            st.error(f"❌ Gagal membuat peramalan bertingkat: {str(e)}")
            st.error(traceback.format_exc())
            return None
            
    def get_market_summary(self):
        summary = {}
        for asset_name, asset_data in self.data.items():
            if not asset_data.empty and len(asset_data) > 1:
                try:
                    current_price = asset_data['Close'].iloc[-1]
                    previous_price = asset_data['Close'].iloc[-2]
                    daily_change = (current_price - previous_price) / previous_price
                    summary[asset_name] = {'current_price': current_price, 'daily_change': daily_change}
                except IndexError:
                    continue
        return summary

    def get_feature_importance(self):
        importance_data = {}
        if not hasattr(self, 'X_columns') or not self.X_columns:
            return {}
        try:
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    normalized_importances = importances / importances.sum()
                    importance_data[model_name] = sorted(list(zip(self.X_columns, normalized_importances)), key=lambda x: x[1], reverse=True)
        except Exception as e:
            st.warning(f"⚠️ Gagal mendapatkan feature importance: {str(e)}")
        return importance_data
