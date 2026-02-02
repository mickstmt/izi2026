import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sqlalchemy import create_engine
import sys

# Add root to sys path for db_config
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from db_config import get_db_url
except ImportError:
    def get_db_url(): return None

class DataProcessor:
    # Constantes de ROAS esperado por plataforma
    ROAS_TARGETS = {
        'META': 12.0,
        'TIKTOK': 8.0
    }

    def __init__(self, excel_path=None):
        self.excel_path = excel_path
        self.df = None
        self.df_ads = None  # DataFrame para datos de publicidad
        self.historical_stats = {}
        self.use_db = False

    def load_from_db(self):
        url = get_db_url()
        if not url:
            return False
        try:
            engine = create_engine(url)
            # Verificamos si la tabla existe leyendo un poco
            df = pd.read_sql("SELECT * FROM ventas_master", engine)
            if not df.empty:
                df['fecha'] = pd.to_datetime(df['fecha'])
                # Mapeamos internamente a los nombres que usa el resto del código
                df['Fecha'] = df['fecha']
                df['Producto'] = df['producto']
                df['Precio Venta'] = df['venta']
                df['Costo Producto'] = df['costo']
                df['Canal'] = df['plataforma']
                df['Categoria'] = df.get('categoria', 'Otros')
                df['Margen'] = df['Precio Venta'] - df['Costo Producto']
                df['Gasto Ads'] = 0
                
                self.df = df
                self.use_db = True
                print("Datos cargados exitosamente desde PostgreSQL (ventas_master)")
                return True
        except Exception as e:
            print(f"PostgreSQL no disponible o tabla no creada: {e}")
        return False

    def load_ads_from_db(self):
        """Carga datos de publicidad desde PostgreSQL"""
        url = get_db_url()
        if not url:
            return False
        try:
            import urllib.parse
            # Decodificar contraseña si está URL-encoded
            from db_config import DB_CONFIG
            password = urllib.parse.unquote(DB_CONFIG['password'])

            # Re-encodear la contraseña para la URL de conexión
            password_encoded = urllib.parse.quote_plus(password)

            # Crear engine con contraseña correctamente encoded para URL
            url_decoded = f"postgresql://{DB_CONFIG['user']}:{password_encoded}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            engine = create_engine(url_decoded)

            df_ads = pd.read_sql("SELECT * FROM ads_costs ORDER BY fecha", engine)
            if not df_ads.empty:
                df_ads['fecha'] = pd.to_datetime(df_ads['fecha'])
                self.df_ads = df_ads
                print(f"Datos de publicidad cargados desde PostgreSQL: {len(df_ads)} registros")
                return True
        except Exception as e:
            print(f"No se pudieron cargar datos de ads desde PostgreSQL: {e}")
        return False

    def load_ads_from_excel(self, ads_excel_path=None):
        """Carga datos de publicidad desde Excel como fallback"""
        if ads_excel_path is None:
            # Buscar archivo por defecto
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ads_excel_path = os.path.join(base_dir, 'data', 'costos-ads.xlsx')

        if not os.path.exists(ads_excel_path):
            print(f"Archivo de ads no encontrado: {ads_excel_path}")
            return False

        try:
            df_ads = pd.read_excel(ads_excel_path)

            # Mapeo de meses en español a números
            meses_esp = {
                'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
                'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
                'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
            }

            # Renombrar columnas
            df_ads.columns = ['plataforma', 'anio', 'mes_nombre', 'gasto_soles', 'venta_soles', 'roas']

            # Convertir mes a número
            df_ads['mes'] = df_ads['mes_nombre'].str.lower().map(meses_esp)

            # Crear fecha
            df_ads['fecha'] = pd.to_datetime(df_ads[['anio', 'mes']].assign(dia=1).rename(
                columns={'anio': 'year', 'mes': 'month', 'dia': 'day'}
            ))

            # Limpiar datos
            df_ads['gasto_soles'] = df_ads['gasto_soles'].fillna(0)
            df_ads['venta_soles'] = df_ads['venta_soles'].fillna(0)
            df_ads['roas'] = df_ads['roas'].fillna(0)

            self.df_ads = df_ads.sort_values('fecha')
            print(f"Datos de publicidad cargados desde Excel: {len(df_ads)} registros")
            return True

        except Exception as e:
            print(f"Error al cargar ads desde Excel: {e}")
            return False

    def load_ads_data(self):
        """Carga datos de publicidad desde BD o Excel"""
        if self.load_ads_from_db():
            return True
        return self.load_ads_from_excel()

    def load_data(self):
        # Primero intentamos desde la DB
        if self.load_from_db():
            # También cargar datos de ads
            self.load_ads_data()
            return True
            
        if not self.excel_path or not os.path.exists(self.excel_path):
            # Generate dummy data if no file is provided for demo purposes
            return self._generate_dummy_data()
        
        try:
            df = pd.read_excel(self.excel_path)
            
            # Mapping columns based on the provided export format
            column_mapping = {
                'FechaVenta': 'Fecha',
                'DescripcionProducto': 'Producto',
                'PrecioVentaConIGV_PEN': 'Precio Venta',
                'CostoProductoPEN': 'Costo Producto',
                'Plataforma': 'Canal'
            }
            
            # Diagnostic: capture original columns
            original_cols = df.columns.tolist()

            # Renaming and ensuring data types
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = df[old_col]
                else:
                    print(f"Warning: Column {old_col} not found in Excel.")
            
            # Diagnostic: check if columns were found
            print(f"Columns in Excel: {df.columns.tolist()}")

            # Basic cleaning
            if 'Fecha' in df.columns:
                df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
                df = df.dropna(subset=['Fecha']) # Remove rows without date
            else:
                # Fallback to any column that looks like a date if FechaVenta is missing
                date_cols = [c for c in df.columns if 'Fecha' in c]
                if date_cols:
                    df['Fecha'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                    df = df.dropna(subset=['Fecha'])
                else:
                    raise KeyError("No se encontró la columna de Fecha")
            
            # Ensure price and cost are numeric
            df['Precio Venta'] = pd.to_numeric(df['Precio Venta'], errors='coerce').fillna(0)
            df['Costo Producto'] = pd.to_numeric(df['Costo Producto'], errors='coerce').fillna(0)
            
            # Calculate Margin
            df['Margen'] = df['Precio Venta'] - df['Costo Producto']
            
            # Filter out years that are clearly wrong (e.g. year 1)
            df = df[df['Fecha'].dt.year > 2000]

            self.df = df
            print(f"Data loaded successfully. Rows: {len(df)}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _generate_dummy_data(self):
        # Create dummy data for 2025
        dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
        data = {
            'Fecha': np.random.choice(dates, 1000),
            'Producto': np.random.choice(['Correa Sport', 'Case Silicone', 'Case Leather', 'Correa Metal', 'Screen Protector'], 1000),
            'Precio Venta': np.random.uniform(50, 150, 1000),
            'Costo Producto': np.random.uniform(20, 60, 1000),
            'Canal': np.random.choice(['Web', 'WhatsApp'], 1000),
            'Gasto Ads': np.random.uniform(5, 15, 1000)
        }
        df = pd.DataFrame(data)
        df['Margen'] = df['Precio Venta'] - df['Costo Producto']
        self.df = df
        return True

    def get_historical_metrics(self, base_year=2025):
        if self.df is None:
            return {}
        
        # Filter data for the selected base year
        if base_year == 'all':
            df_base = self.df.copy()
        else:
            base_year = int(base_year)
            df_base = self.df[self.df['Fecha'].dt.year == base_year].copy()
        
        if df_base.empty:
            # Fallback: use the most recent year
            latest_year = self.df['Fecha'].dt.year.max()
            df_base = self.df[self.df['Fecha'].dt.year == latest_year].copy()
            base_year = latest_year

        total_sales = df_base['Precio Venta'].sum()
        total_costs = df_base['Costo Producto'].sum()
        
        # Avoid division by zero
        margin_ratio = (total_sales - total_costs) / total_sales if total_sales > 0 else 0
        cost_ratio = total_costs / total_sales if total_sales > 0 else 0
        
        avg_monthly_sales = df_base.groupby(df_base['Fecha'].dt.month)['Precio Venta'].sum().mean() if not df_base.empty else 0
        
        self.historical_stats = {
            'margin_ratio': margin_ratio,
            'cost_ratio': cost_ratio,
            'avg_monthly_sales': avg_monthly_sales,
            'total_sales_base': total_sales,
            'base_year': base_year
        }
        return self.historical_stats

    def get_platform_proportions(self, base_year=2025):
        """
        Calcula proporción histórica de ventas por plataforma de publicidad.

        Args:
            base_year: Año para analizar proporciones (o 'all' para todos los años)

        Returns:
            dict: {'META': 0.XX, 'TIKTOK': 0.XX} representando proporción de ventas totales de ads
        """
        if self.df_ads is None or self.df_ads.empty:
            # No hay datos de ads - retornar split igual como fallback
            return {'META': 0.5, 'TIKTOK': 0.5}

        # Filtrar por año base
        if base_year == 'all':
            df_year = self.df_ads.copy()
        else:
            base_year = int(base_year)
            df_year = self.df_ads[self.df_ads['anio'] == base_year]

        if df_year.empty:
            # No hay datos para este año - usar todos los datos disponibles
            df_year = self.df_ads.copy()

        # Agrupar por plataforma y sumar ventas
        platform_sales = df_year.groupby('plataforma')['venta_soles'].sum()
        total_sales = platform_sales.sum()

        # Manejar caso edge: no hay ventas
        if total_sales == 0:
            return {'META': 0.5, 'TIKTOK': 0.5}

        # Calcular proporciones
        proportions = {}
        for platform in ['META', 'TIKTOK']:
            if platform in platform_sales.index:
                proportions[platform] = float(platform_sales[platform] / total_sales)
            else:
                proportions[platform] = 0.0

        # Asegurar que las proporciones sumen 1.0 (manejar caso donde solo existe una plataforma)
        if proportions['META'] == 0 and proportions['TIKTOK'] == 0:
            proportions = {'META': 0.5, 'TIKTOK': 0.5}
        elif proportions['META'] == 0:
            proportions['META'] = 0.01  # Dar al menos 1% para evitar problemas de división
            proportions['TIKTOK'] = 0.99
        elif proportions['TIKTOK'] == 0:
            proportions['TIKTOK'] = 0.01
            proportions['META'] = 0.99

        return proportions

    def get_yearly_history_chart(self):
        if self.df is None or self.df.empty:
            return None
        
        # Group by year
        df_copy = self.df.copy()
        df_copy['Year'] = df_copy['Fecha'].dt.year
        
        yearly_data = df_copy.groupby('Year').agg({
            'Precio Venta': 'sum',
            'Margen': 'sum'
        }).reset_index()
        
        # Sort by year to ensure correct order
        yearly_data = yearly_data.sort_values('Year')
        
        # Diagnostic print to server console
        print("Historical Yearly Data Summary:")
        print(yearly_data)
        
        years = yearly_data['Year'].astype(str).tolist()
        ventas = yearly_data['Precio Venta'].tolist()
        margen = yearly_data['Margen'].tolist()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=years, 
            y=ventas, 
            name='Ventas Totales', 
            marker_color='#3498db',
            text=[f"S/ {v:,.0f}" for v in ventas],
            textposition='auto',
            orientation='v'
        ))
        
        fig.add_trace(go.Bar(
            x=years, 
            y=margen, 
            name='Margen Bruto', 
            marker_color='#2ecc71',
            text=[f"S/ {v:,.0f}" for v in margen],
            textposition='auto',
            orientation='v'
        ))
        
        fig.update_layout(
            title='Evolución Anual: Ventas vs Margen',
            xaxis=dict(title='Año', type='category'),
            yaxis=dict(title='Monto (S/)', tickformat=',.2f'),
            barmode='group',
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig.to_json()

    def get_historical_monthly_trends(self):
        if self.df is None or self.df.empty:
            return None
        
        # Get monthly data for each year - work with original column names from DB
        df_copy = self.df.copy()
        
        # Ensure we have the year and month
        if 'Fecha' not in df_copy.columns and 'fecha' in df_copy.columns:
            df_copy['Fecha'] = df_copy['fecha']
        
        df_copy['Year'] = df_copy['Fecha'].dt.year
        df_copy['Month'] = df_copy['Fecha'].dt.month
        
        # Determine which sales column to use
        if 'venta' in df_copy.columns:
            sales_col = 'venta'
        elif 'Precio Venta' in df_copy.columns:
            sales_col = 'Precio Venta'
        else:
            return None
        
        monthly_data = df_copy.groupby(['Year', 'Month'])[sales_col].sum().reset_index()
        monthly_data.columns = ['Year', 'Month', 'Ventas']
        
        # Debug print
        print(f"Monthly data sample: {monthly_data.head(10).to_dict('records')}")
        
        fig = go.Figure()
        
        # Create a line for each year
        for year in sorted(monthly_data['Year'].unique()):
            year_data = monthly_data[monthly_data['Year'] == year].sort_values('Month')
            month_names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            months = [month_names[int(m)-1] for m in year_data['Month'].values]
            sales_values = year_data['Ventas'].values.tolist()
            
            fig.add_trace(go.Scatter(
                x=months,
                y=sales_values,
                name=str(int(year)),
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>{int(year)} - %{{x}}</b><br>S/ %{{y:,.0f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Evolución Mensual Real por Año (Histórico)',
            xaxis=dict(title='Mes', type='category'),
            yaxis=dict(title='Ventas (S/)', tickformat=',.0f'),
            template='plotly_white',
            hovermode='x unified',
            legend=dict(title='Año', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig.to_json()

    def run_projection(self, growth_pct, roas_meta=12.0, roas_tiktok=8.0, base_year=2025):
        # Validar inputs de ROAS
        roas_meta = max(0.1, float(roas_meta))
        roas_tiktok = max(0.1, float(roas_tiktok))

        stats = self.get_historical_metrics(base_year)
        platform_props = self.get_platform_proportions(base_year)

        # Base data for seasonality
        if base_year == 'all':
            df_base = self.df.copy()
        else:
            base_year = int(stats['base_year'])
            df_base = self.df[self.df['Fecha'].dt.year == base_year]

        monthly_base = df_base.groupby(df_base['Fecha'].dt.month)['Precio Venta'].sum()

        months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        projections = []

        for i, month_name in enumerate(months):
            month_idx = i + 1
            base_sales = monthly_base.get(month_idx, stats['avg_monthly_sales'])

            # Aplicar crecimiento para obtener ventas proyectadas totales
            projected_sales = base_sales * (1 + growth_pct / 100)

            # Dividir ventas por plataforma usando proporciones históricas
            projected_sales_meta = projected_sales * platform_props['META']
            projected_sales_tiktok = projected_sales * platform_props['TIKTOK']

            # Calcular gasto en ads requerido basado en ROAS targets
            # ROAS = Ventas / Gasto, por lo tanto Gasto = Ventas / ROAS
            projected_ads_meta = projected_sales_meta / roas_meta if roas_meta > 0 else 0
            projected_ads_tiktok = projected_sales_tiktok / roas_tiktok if roas_tiktok > 0 else 0
            projected_ads_total = projected_ads_meta + projected_ads_tiktok

            # Calcular costos y utilidad
            projected_cogs = projected_sales * stats['cost_ratio']
            net_profit = projected_sales - projected_cogs - projected_ads_total

            projections.append({
                'Mes': month_name,
                'Ventas': projected_sales,
                'Costos Mercaderia': projected_cogs,
                'Gasto Ads': projected_ads_total,
                'Gasto Ads META': projected_ads_meta,
                'Gasto Ads TIKTOK': projected_ads_tiktok,
                'Utilidad Neta': net_profit
            })

        return pd.DataFrame(projections)

    def create_trend_chart(self, proj_df):
        fig = go.Figure()
        
        # Convert to lists to avoid Plotly-Pandas index confusion
        meses = proj_df['Mes'].tolist()
        ventas = proj_df['Ventas'].tolist()
        utilidad = proj_df['Utilidad Neta'].tolist()
        costos_totales = (proj_df['Costos Mercaderia'] + proj_df['Gasto Ads']).tolist()

        # Add Utility as Bars
        fig.add_trace(go.Bar(
            x=meses, 
            y=utilidad,
            name='Utilidad Neta',
            marker_color='rgba(46, 204, 113, 0.4)',
            orientation='v',
            hovertemplate='Utilidad %{x}: S/ %{y:,.2f}<extra></extra>'
        ))
        
        # Add Sales as Line
        fig.add_trace(go.Scatter(
            x=meses, 
            y=ventas,
            name='Ventas',
            mode='lines+markers',
            line=dict(color='#2980b9', width=4),
            hovertemplate='Ventas %{x}: S/ %{y:,.2f}<extra></extra>'
        ))
        
        # Add Total Costs (Product + Ads) as Line
        fig.add_trace(go.Scatter(
            x=meses, 
            y=costos_totales,
            name='Costos Totales',
            mode='lines+markers',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            hovertemplate='Costos %{x}: S/ %{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Proyección Mensual de Flujo de Caja 2026',
            xaxis=dict(title='Mes', type='category'),
            yaxis=dict(title='Monto (S/)', tickformat=',.2f'),
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig.to_json()

    def create_composition_chart(self, proj_df):
        total_sales = proj_df['Ventas'].sum()
        total_cogs = proj_df['Costos Mercaderia'].sum()
        total_ads = proj_df['Gasto Ads'].sum()
        total_profit = proj_df['Utilidad Neta'].sum()
        
        labels = ['Costo Mercadería', 'Gasto Ads', 'Utilidad Neta']
        values = [total_cogs, total_ads, total_profit]
        
        fig = px.pie(names=labels, values=values, title='Distribución Proyectada 2026', hole=.4)
        fig.update_traces(textinfo='percent+label')
        
        return fig.to_json()

    def create_trend_based_composition_chart(self):
        """
        Crea gráfico de composición basado en tendencias históricas
        En lugar de usar los sliders, calcula automáticamente basándose en el promedio histórico

        Returns:
            JSON del gráfico o None
        """
        trend_proj = self.calculate_trend_based_projection_2026()

        if not trend_proj:
            return None

        labels = ['Costo Mercadería', 'Gasto Ads', 'Utilidad Neta']
        values = [
            trend_proj['projected_costo'],
            trend_proj['projected_ads'],
            trend_proj['projected_utilidad']
        ]

        # Crear título con información de tendencias
        title_text = (
            f"Distribución Proyectada 2026<br>"
            f"<sub>Basada en tendencias históricas | "
            f"Ventas proyectadas: S/ {trend_proj['projected_sales']:,.0f} "
            f"({trend_proj['growth_from_2025']:+.1f}%)</sub>"
        )

        fig = px.pie(
            names=labels,
            values=values,
            title=title_text,
            hole=.4,
            color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71']
        )

        fig.update_traces(
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Monto: S/ %{value:,.2f}<br>Porcentaje: %{percent}<extra></extra>'
        )

        # Añadir anotación con promedios históricos
        fig.add_annotation(
            text=f"Promedios históricos:<br>"
                 f"Costos: {trend_proj['avg_costo_pct']:.1f}%<br>"
                 f"Ads: {trend_proj['avg_ads_pct']:.1f}%<br>"
                 f"Margen: {trend_proj['avg_margen_pct']:.1f}%",
            xref="paper", yref="paper",
            x=1.15, y=0.5,
            showarrow=False,
            font=dict(size=10),
            align="left"
        )

        return fig.to_json()

    def get_category_analysis(self):
        if self.df is None or 'Categoria' not in self.df.columns:
            return None

        # Calcular totales por categoría
        cat_data = self.df.groupby('Categoria').agg({
            'Precio Venta': 'sum',
            'Margen': 'sum'
        }).reset_index()

        # Calcular margen porcentual por categoría
        cat_data['Margen %'] = (cat_data['Margen'] / cat_data['Precio Venta'] * 100)

        # Ordenar por margen % descendente
        cat_data = cat_data.sort_values('Margen %', ascending=False)

        # Convertir a listas para evitar problemas con índices
        categorias = cat_data['Categoria'].tolist()
        margenes = cat_data['Margen %'].tolist()
        ventas = cat_data['Precio Venta'].tolist()
        utilidades = cat_data['Margen'].tolist()

        # Crear gráfico de barras
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=categorias,
            y=margenes,
            text=[f"{val:.1f}%" for val in margenes],
            textposition='outside',
            marker_color='#2ecc71',
            hovertemplate='<b>%{x}</b><br>Margen: %{y:.1f}%<br>Ventas: S/ %{customdata[0]:,.0f}<br>Utilidad: S/ %{customdata[1]:,.0f}<extra></extra>',
            customdata=list(zip(ventas, utilidades))
        ))

        fig.update_layout(
            title='Rentabilidad por Categoría de Producto (Margen %)',
            xaxis=dict(title='Categoría'),
            yaxis=dict(
                title='Margen (%)',
                range=[0, 90]
            ),
            template='plotly_white',
            showlegend=False,
            height=400
        )

        return fig.to_json()

    def get_top_products(self):
        try:
            if self.df is None or self.df.empty:
                print("ERROR: DataFrame is None or empty in get_top_products")
                return None
            
            print(f"DataFrame columns: {self.df.columns.tolist()}")
            print(f"DataFrame shape: {self.df.shape}")
            
            # Asegurar que tenemos las columnas correctas
            df_copy = self.df.copy()
            
            # Calcular margen si no existe
            if 'Margen' not in df_copy.columns:
                if 'margen_unitario' in df_copy.columns:
                    df_copy['Margen'] = df_copy['margen_unitario']
                elif 'Precio Venta' in df_copy.columns and 'Costo Producto' in df_copy.columns:
                    df_copy['Margen'] = df_copy['Precio Venta'] - df_copy['Costo Producto']
                else:
                    print(f"ERROR: No se puede calcular Margen. Columnas: {df_copy.columns.tolist()}")
                    return None
            
            top_products = df_copy.groupby('Producto').agg({
                'Margen': 'sum'
            }).sort_values('Margen', ascending=False).head(5).reset_index()
            
            # Asegurar que Margen sea numérico
            top_products['Margen'] = pd.to_numeric(top_products['Margen'], errors='coerce')
            
            print(f"Top products data: {top_products.to_dict('records')}")
            print(f"Margen values: {top_products['Margen'].tolist()}")
            print(f"Margen dtype: {top_products['Margen'].dtype}")
            
            # Crear gráfico simple
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_products['Producto'].tolist(),
                y=top_products['Margen'].tolist(),
                text=[f"S/ {val:,.0f}" for val in top_products['Margen'].tolist()],
                textposition='outside',
                marker_color='royalblue',
                hovertemplate='<b>%{x}</b><br>Margen: S/ %{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Top 5 Productos por Margen de Contribución',
                xaxis_title='Producto',
                yaxis_title='Margen Total (S/)',
                yaxis=dict(tickformat=',.0f'),
                template='plotly_white',
                showlegend=False
            )
            
            return fig.to_json()
        except Exception as e:
            print(f"ERROR in get_top_products: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    # ============================================================
    # MÉTODOS PARA ANÁLISIS DE PUBLICIDAD
    # ============================================================

    def calculate_ads_metrics(self, year=None, platform=None):
        """
        Calcula métricas agregadas de publicidad

        Args:
            year: Año específico o None para todos
            platform: Plataforma específica ('META', 'TIKTOK') o None para todas

        Returns:
            dict con métricas calculadas
        """
        if self.df_ads is None or self.df_ads.empty:
            return {
                'total_gasto': 0,
                'total_venta': 0,
                'roas_promedio': 0,
                'mejor_mes': None,
                'peor_mes': None
            }

        df = self.df_ads.copy()

        # Filtrar por año si se especifica
        if year is not None:
            df = df[df['anio'] == year]

        # Filtrar por plataforma si se especifica
        if platform is not None:
            df = df[df['plataforma'] == platform]

        if df.empty:
            return {
                'total_gasto': 0,
                'total_venta': 0,
                'roas_promedio': 0,
                'mejor_mes': None,
                'peor_mes': None
            }

        total_gasto = df['gasto_soles'].sum()
        total_venta = df['venta_soles'].sum()
        roas_promedio = (total_venta / total_gasto) if total_gasto > 0 else 0

        # Mejor y peor mes por ROAS
        df_filtered = df[df['roas'] > 0]
        mejor_mes = None
        peor_mes = None

        if not df_filtered.empty:
            idx_mejor = df_filtered['roas'].idxmax()
            idx_peor = df_filtered['roas'].idxmin()

            mejor_mes = {
                'fecha': df_filtered.loc[idx_mejor, 'fecha'],
                'plataforma': df_filtered.loc[idx_mejor, 'plataforma'],
                'roas': df_filtered.loc[idx_mejor, 'roas'],
                'gasto': df_filtered.loc[idx_mejor, 'gasto_soles'],
                'venta': df_filtered.loc[idx_mejor, 'venta_soles']
            }

            peor_mes = {
                'fecha': df_filtered.loc[idx_peor, 'fecha'],
                'plataforma': df_filtered.loc[idx_peor, 'plataforma'],
                'roas': df_filtered.loc[idx_peor, 'roas'],
                'gasto': df_filtered.loc[idx_peor, 'gasto_soles'],
                'venta': df_filtered.loc[idx_peor, 'venta_soles']
            }

        return {
            'total_gasto': float(total_gasto),
            'total_venta': float(total_venta),
            'roas_promedio': float(roas_promedio),
            'mejor_mes': mejor_mes,
            'peor_mes': peor_mes,
            'num_registros': len(df)
        }

    def calculate_ads_goals(self, target_year, growth_pct=10):
        """
        Calcula metas de venta y gasto en publicidad basadas en histórico

        Args:
            target_year: Año objetivo para la meta
            growth_pct: Porcentaje de crecimiento esperado sobre el promedio histórico

        Returns:
            dict con metas por plataforma
        """
        if self.df_ads is None or self.df_ads.empty:
            return {}

        # Filtrar datos históricos (años anteriores al target_year)
        df_historical = self.df_ads[self.df_ads['anio'] < target_year]

        if df_historical.empty:
            return {}

        goals = {}

        for platform in df_historical['plataforma'].unique():
            df_platform = df_historical[df_historical['plataforma'] == platform]

            # Calcular promedios mensuales históricos
            avg_monthly_gasto = df_platform['gasto_soles'].mean()
            avg_monthly_venta = df_platform['venta_soles'].mean()
            avg_roas = (df_platform['venta_soles'].sum() / df_platform['gasto_soles'].sum()) if df_platform['gasto_soles'].sum() > 0 else 0

            # Aplicar crecimiento
            meta_venta_mensual = avg_monthly_venta * (1 + growth_pct / 100)
            meta_gasto_mensual = avg_monthly_gasto * (1 + growth_pct / 100)

            # Metas anuales
            meta_venta_anual = meta_venta_mensual * 12
            meta_gasto_anual = meta_gasto_mensual * 12

            # ROAS esperado (de la constante)
            roas_esperado = self.ROAS_TARGETS.get(platform, avg_roas)

            goals[platform] = {
                'meta_venta_mensual': float(meta_venta_mensual),
                'meta_gasto_mensual': float(meta_gasto_mensual),
                'meta_venta_anual': float(meta_venta_anual),
                'meta_gasto_anual': float(meta_gasto_anual),
                'roas_esperado': float(roas_esperado),
                'roas_historico': float(avg_roas),
                'crecimiento_aplicado': growth_pct
            }

        return goals

    def get_ads_comparison_data(self, year, roas_meta=None, roas_tiktok=None):
        """
        Obtiene datos para comparar Venta Real vs Meta y ROAS Real vs Esperado

        Args:
            year: Año para el análisis
            roas_meta: ROAS esperado para META (opcional, usa ROAS_TARGETS si no se especifica)
            roas_tiktok: ROAS esperado para TIKTOK (opcional, usa ROAS_TARGETS si no se especifica)

        Returns:
            dict con datos de comparación
        """
        if self.df_ads is None or self.df_ads.empty:
            return {}

        # Usar ROAS configurados o valores por defecto
        roas_targets = {
            'META': roas_meta if roas_meta is not None else self.ROAS_TARGETS.get('META', 12.0),
            'TIKTOK': roas_tiktok if roas_tiktok is not None else self.ROAS_TARGETS.get('TIKTOK', 8.0)
        }

        # Obtener datos reales del año
        df_year = self.df_ads[self.df_ads['anio'] == year]

        # Calcular metas basadas en años anteriores
        goals = self.calculate_ads_goals(year, growth_pct=10)

        comparison = {}

        for platform in df_year['plataforma'].unique():
            df_platform = df_year[df_year['plataforma'] == platform]

            venta_real = df_platform['venta_soles'].sum()
            gasto_real = df_platform['gasto_soles'].sum()
            roas_real = (venta_real / gasto_real) if gasto_real > 0 else 0

            # Obtener metas
            goal_data = goals.get(platform, {})
            meta_venta = goal_data.get('meta_venta_anual', venta_real)
            roas_esperado = roas_targets.get(platform, 10)

            # Calcular cumplimiento
            cumplimiento_venta = (venta_real / meta_venta * 100) if meta_venta > 0 else 0
            cumplimiento_roas = (roas_real / roas_esperado * 100) if roas_esperado > 0 else 0

            comparison[platform] = {
                'venta_real': float(venta_real),
                'meta_venta': float(meta_venta),
                'cumplimiento_venta_pct': float(cumplimiento_venta),
                'roas_real': float(roas_real),
                'roas_esperado': float(roas_esperado),
                'cumplimiento_roas_pct': float(cumplimiento_roas),
                'gasto_real': float(gasto_real)
            }

        return comparison

    def create_ads_spending_chart(self):
        """Crea gráfico de gasto en publicidad por plataforma y mes"""
        if self.df_ads is None or self.df_ads.empty:
            return None

        df = self.df_ads.copy()

        # Agrupar por fecha y plataforma para eliminar duplicados y sumar gastos del mismo mes
        df_grouped = df.groupby(['fecha', 'plataforma']).agg({
            'gasto_soles': 'sum',
            'venta_soles': 'sum',
            'roas': 'mean'
        }).reset_index()

        df_grouped['fecha_str'] = df_grouped['fecha'].dt.strftime('%Y-%m')

        fig = go.Figure()

        for platform in sorted(df_grouped['plataforma'].unique()):
            df_platform = df_grouped[df_grouped['plataforma'] == platform].sort_values('fecha')

            # Convertir a listas para evitar problemas de renderizado con Plotly
            x_values = df_platform['fecha_str'].tolist()
            y_values = df_platform['gasto_soles'].tolist()

            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                name=platform,
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>{platform}</b><br>%{{x}}<br>Gasto: S/ %{{y:,.2f}}<extra></extra>'
            ))

        # Calcular rango del eje Y basado en los datos reales
        max_value = float(df_grouped['gasto_soles'].max())
        y_range = [0, max_value * 1.05]  # Agregar solo 5% de margen superior para mejor uso del espacio

        fig.update_layout(
            title='Inversión en Publicidad por Plataforma',
            autosize=True,  # Permitir que Plotly maneje el tamaño
            height=400,
            xaxis=dict(title='Mes', tickangle=-45),
            yaxis=dict(
                title='Gasto (S/)',
                tickformat=',.0f',
                range=y_range
            ),
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig.to_json()

    def create_roas_comparison_chart(self, year=2025, roas_meta=None, roas_tiktok=None):
        """Crea gráfico comparativo de ROAS Real vs Esperado"""
        comparison = self.get_ads_comparison_data(year, roas_meta, roas_tiktok)

        if not comparison:
            return None

        platforms = list(comparison.keys())
        roas_real = [comparison[p]['roas_real'] for p in platforms]
        roas_esperado = [comparison[p]['roas_esperado'] for p in platforms]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='ROAS Real',
            x=platforms,
            y=roas_real,
            marker_color='#3498db',
            text=[f"{r:.2f}x" for r in roas_real],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>ROAS Real: %{y:.2f}x<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='ROAS Esperado',
            x=platforms,
            y=roas_esperado,
            marker_color='#95a5a6',
            text=[f"{r:.2f}x" for r in roas_esperado],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>ROAS Esperado: %{y:.2f}x<extra></extra>'
        ))

        fig.update_layout(
            title=f'ROAS Real vs Esperado - {year}',
            xaxis=dict(title='Plataforma'),
            yaxis=dict(title='ROAS', tickformat='.2f'),
            template='plotly_white',
            barmode='group',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig.to_json()

    def create_sales_goal_comparison_chart(self, year=2025):
        """Crea gráfico comparativo de Venta Real vs Meta"""
        comparison = self.get_ads_comparison_data(year)

        if not comparison:
            return None

        platforms = list(comparison.keys())
        venta_real = [comparison[p]['venta_real'] for p in platforms]
        meta_venta = [comparison[p]['meta_venta'] for p in platforms]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Venta Real',
            x=platforms,
            y=venta_real,
            marker_color='#2ecc71',
            text=[f"S/ {v:,.0f}" for v in venta_real],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Venta Real: S/ %{y:,.2f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='Meta de Venta',
            x=platforms,
            y=meta_venta,
            marker_color='#e74c3c',
            text=[f"S/ {v:,.0f}" for v in meta_venta],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Meta: S/ %{y:,.2f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Venta Real vs Meta - {year}',
            xaxis=dict(title='Plataforma'),
            yaxis=dict(title='Ventas (S/)', tickformat=',.0f'),
            template='plotly_white',
            barmode='group',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig.to_json()

    def get_ads_attribution_analysis(self, year=None):
        """
        Analiza qué porcentaje de las ventas totales vienen de publicidad pagada

        Args:
            year: Año específico o None para todos

        Returns:
            dict con análisis de atribución
        """
        if self.df is None or self.df_ads is None:
            return {}

        # Filtrar por año si se especifica
        if year is not None:
            df_ventas = self.df[self.df['Fecha'].dt.year == year]
            df_ads = self.df_ads[self.df_ads['anio'] == year]
        else:
            df_ventas = self.df
            df_ads = self.df_ads

        # Ventas totales
        ventas_totales = df_ventas['Precio Venta'].sum()

        # Ventas atribuidas a publicidad
        ventas_ads = df_ads['venta_soles'].sum()

        # Ventas orgánicas (no de publicidad)
        ventas_organicas = ventas_totales - ventas_ads

        # Porcentajes
        pct_ads = (ventas_ads / ventas_totales * 100) if ventas_totales > 0 else 0
        pct_organicas = (ventas_organicas / ventas_totales * 100) if ventas_totales > 0 else 0

        return {
            'ventas_totales': float(ventas_totales),
            'ventas_ads': float(ventas_ads),
            'ventas_organicas': float(ventas_organicas),
            'pct_ads': float(pct_ads),
            'pct_organicas': float(pct_organicas),
            'gasto_ads_total': float(df_ads['gasto_soles'].sum())
        }

    def create_monthly_roas_trend(self, year=2025):
        """
        Crea gráfico de línea mostrando ROAS mes a mes por plataforma

        Args:
            year: Año para analizar

        Returns:
            JSON del gráfico o None
        """
        if self.df_ads is None or self.df_ads.empty:
            return None

        # Filtrar por año
        df_year = self.df_ads[self.df_ads['anio'] == year].copy()

        if df_year.empty:
            return None

        # Convertir mes_nombre a número para ordenar correctamente
        meses_orden = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        df_year['mes_num'] = df_year['mes_nombre'].map(meses_orden)

        # Calcular ROAS por mes y plataforma
        monthly_roas = df_year.groupby(['plataforma', 'mes_nombre', 'mes_num']).agg({
            'venta_soles': 'sum',
            'gasto_soles': 'sum'
        }).reset_index()

        monthly_roas['roas'] = monthly_roas.apply(
            lambda row: row['venta_soles'] / row['gasto_soles'] if row['gasto_soles'] > 0 else 0,
            axis=1
        )

        # Ordenar por mes
        monthly_roas = monthly_roas.sort_values('mes_num')

        # Crear lista completa de todos los meses en orden
        all_months = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                      'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']

        fig = go.Figure()

        # Crear línea para cada plataforma
        colors = {'META': '#3498db', 'TIKTOK': '#e74c3c'}
        for platform in ['META', 'TIKTOK']:  # Asegurar orden consistente
            df_platform = monthly_roas[monthly_roas['plataforma'] == platform]

            # Crear diccionario de mes -> roas
            roas_dict = dict(zip(df_platform['mes_nombre'], df_platform['roas']))

            # Llenar todos los meses con None si no tienen datos
            roas_values = [roas_dict.get(month, None) for month in all_months]

            fig.add_trace(go.Scatter(
                x=all_months,
                y=roas_values,
                name=platform,
                mode='lines+markers',
                line=dict(color=colors.get(platform, '#95a5a6'), width=3),
                marker=dict(size=8),
                connectgaps=False,  # No conectar gaps donde no hay datos
                hovertemplate='<b>%{x}</b><br>' + platform + ' ROAS: %{y:.2f}x<extra></extra>'
            ))

        fig.update_layout(
            title=f'ROAS Histórico Mes a Mes - {year}',
            xaxis=dict(title='Mes', type='category', categoryorder='array', categoryarray=all_months),
            yaxis=dict(title='ROAS', tickformat='.1f'),
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig.to_json()

    def create_monthly_ads_spending(self, year=2025):
        """
        Crea gráfico de barras apiladas mostrando inversión mensual por plataforma

        Args:
            year: Año para analizar

        Returns:
            JSON del gráfico o None
        """
        if self.df_ads is None or self.df_ads.empty:
            return None

        # Filtrar por año
        df_year = self.df_ads[self.df_ads['anio'] == year].copy()

        if df_year.empty:
            return None

        # Convertir mes_nombre a número para ordenar correctamente
        meses_orden = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        df_year['mes_num'] = df_year['mes_nombre'].map(meses_orden)

        # Calcular gasto por mes y plataforma
        monthly_spending = df_year.groupby(['plataforma', 'mes_nombre', 'mes_num']).agg({
            'gasto_soles': 'sum'
        }).reset_index()

        # Ordenar por mes
        monthly_spending = monthly_spending.sort_values('mes_num')

        # Crear lista completa de todos los meses en orden
        all_months = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
                      'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']

        fig = go.Figure()

        # Crear barra apilada para cada plataforma
        colors = {'META': '#3498db', 'TIKTOK': '#e74c3c'}
        for platform in ['META', 'TIKTOK']:  # Asegurar orden consistente
            df_platform = monthly_spending[monthly_spending['plataforma'] == platform]

            # Crear diccionario de mes -> gasto
            gasto_dict = dict(zip(df_platform['mes_nombre'], df_platform['gasto_soles']))

            # Llenar todos los meses con 0 si no tienen datos
            gasto_values = [gasto_dict.get(month, 0) for month in all_months]

            fig.add_trace(go.Bar(
                name=platform,
                x=all_months,
                y=gasto_values,
                marker_color=colors.get(platform, '#95a5a6'),
                hovertemplate='<b>%{x}</b><br>' + platform + ': S/ %{y:,.0f}<extra></extra>'
            ))

        fig.update_layout(
            title=f'Inversión en Publicidad Mes a Mes - {year}',
            xaxis=dict(title='Mes', type='category', categoryorder='array', categoryarray=all_months),
            yaxis=dict(title='Inversión (S/)', tickformat=',.0f'),
            template='plotly_white',
            barmode='stack',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig.to_json()

    def create_monthly_investment_projection(self, growth_pct=10, roas_meta=12.0, roas_tiktok=8.0, base_year=2025):
        """
        Crea tabla/gráfico de proyección mensual de inversión para 2026

        Args:
            growth_pct: Porcentaje de crecimiento esperado
            roas_meta: ROAS target para META
            roas_tiktok: ROAS target para TIKTOK
            base_year: Año base para proyección

        Returns:
            JSON del gráfico o None
        """
        # Obtener proyección completa
        proj_df = self.run_projection(growth_pct, roas_meta, roas_tiktok, base_year)

        if proj_df is None or proj_df.empty:
            return None

        # Extraer datos mensuales
        meses = proj_df['Mes'].tolist()
        gasto_meta = proj_df['Gasto Ads META'].tolist()
        gasto_tiktok = proj_df['Gasto Ads TIKTOK'].tolist()
        gasto_total = proj_df['Gasto Ads'].tolist()

        fig = go.Figure()

        # Barras apiladas para META y TIKTOK
        fig.add_trace(go.Bar(
            name='META',
            x=meses,
            y=gasto_meta,
            marker_color='#3498db',
            text=[f"S/ {val:,.0f}" for val in gasto_meta],
            textposition='inside',
            hovertemplate='<b>%{x}</b><br>META: S/ %{y:,.0f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='TIKTOK',
            x=meses,
            y=gasto_tiktok,
            marker_color='#e74c3c',
            text=[f"S/ {val:,.0f}" for val in gasto_tiktok],
            textposition='inside',
            hovertemplate='<b>%{x}</b><br>TIKTOK: S/ %{y:,.0f}<extra></extra>'
        ))

        # Línea de gasto total
        fig.add_trace(go.Scatter(
            name='Total',
            x=meses,
            y=gasto_total,
            mode='lines+markers',
            line=dict(color='#2c3e50', width=3, dash='dash'),
            marker=dict(size=8),
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>Total: S/ %{y:,.0f}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Proyección de Inversión Mensual 2026 (ROAS META: {roas_meta:.1f}x, TIKTOK: {roas_tiktok:.1f}x)',
            xaxis=dict(title='Mes', type='category'),
            yaxis=dict(title='Inversión por Plataforma (S/)', tickformat=',.0f'),
            yaxis2=dict(title='Inversión Total (S/)', overlaying='y', side='right', tickformat=',.0f'),
            template='plotly_white',
            barmode='stack',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig.to_json()

    def calculate_2026_projection(self):
        """
        Calcula la proyección de ventas para 2026 basada en tendencia histórica
        Usa regresión lineal y promedio de tasa de crecimiento

        Returns:
            dict con proyecciones y métodos utilizados
        """
        if self.df is None or self.df.empty:
            return None

        # Crear columna de año
        df_copy = self.df.copy()
        df_copy['Year'] = df_copy['Fecha'].dt.year

        # Obtener ventas por año
        yearly_sales = df_copy.groupby('Year')['Precio Venta'].sum().reset_index()
        yearly_sales = yearly_sales.sort_values('Year')

        if len(yearly_sales) < 2:
            return None

        years = yearly_sales['Year'].values
        sales = yearly_sales['Precio Venta'].values

        # Método 1: Regresión Lineal
        # y = mx + b
        coefficients = np.polyfit(years, sales, 1)  # Grado 1 = línea recta
        slope = coefficients[0]
        intercept = coefficients[1]
        projection_2026_linear = slope * 2026 + intercept

        # Método 2: Promedio de Tasa de Crecimiento
        growth_rates = []
        for i in range(1, len(sales)):
            growth = (sales[i] - sales[i-1]) / sales[i-1]
            growth_rates.append(growth)

        avg_growth_rate = np.mean(growth_rates)
        last_year_sales = sales[-1]
        projection_2026_avg_growth = last_year_sales * (1 + avg_growth_rate)

        # Método 3: Promedio de los dos métodos anteriores
        projection_2026_combined = (projection_2026_linear + projection_2026_avg_growth) / 2

        return {
            'linear_regression': projection_2026_linear,
            'avg_growth_rate': projection_2026_avg_growth,
            'combined': projection_2026_combined,
            'avg_growth_pct': avg_growth_rate * 100,
            'slope': slope,
            'historical_years': years.tolist(),
            'historical_sales': sales.tolist()
        }

    def calculate_trend_based_projection_2026(self):
        """
        Calcula la proyección completa para 2026 basada en tendencias históricas
        Incluye ventas, costos, utilidad, y gastos de publicidad

        Returns:
            dict con proyección completa para 2026
        """
        if self.df is None or self.df.empty:
            return None

        # Obtener proyección de ventas 2026
        sales_projection = self.calculate_2026_projection()
        if not sales_projection:
            return None

        projected_sales_2026 = sales_projection['combined']

        # Calcular promedios históricos
        df_copy = self.df.copy()
        df_copy['Year'] = df_copy['Fecha'].dt.year

        # Filtrar solo años históricos completos (2023, 2024, 2025)
        historical_data = df_copy[df_copy['Year'].isin([2023, 2024, 2025])]

        # Calcular totales por año
        yearly_totals = historical_data.groupby('Year').agg({
            'Precio Venta': 'sum',
            'Costo Producto': 'sum',
            'Margen': 'sum'
        }).reset_index()

        # Calcular porcentajes promedio
        yearly_totals['Costo_Pct'] = yearly_totals['Costo Producto'] / yearly_totals['Precio Venta']
        yearly_totals['Margen_Pct'] = yearly_totals['Margen'] / yearly_totals['Precio Venta']

        avg_costo_pct = yearly_totals['Costo_Pct'].mean()
        avg_margen_pct = yearly_totals['Margen_Pct'].mean()

        # Calcular gasto en publicidad promedio
        avg_ads_pct = 0
        avg_roas = {'META': 12.0, 'TIKTOK': 8.0}  # Defaults

        if self.df_ads is not None and not self.df_ads.empty:
            # Calcular gasto de ads como % de ventas históricas
            ads_by_year = self.df_ads.groupby('anio').agg({
                'gasto_soles': 'sum',
                'venta_soles': 'sum'
            }).reset_index()

            # Unir con ventas totales
            yearly_totals_sales = df_copy.groupby('Year')['Precio Venta'].sum().reset_index()
            yearly_totals_sales.columns = ['Year', 'Total_Ventas']

            ads_by_year.columns = ['Year', 'Gasto_Ads', 'Venta_Ads']
            combined = yearly_totals_sales.merge(ads_by_year, on='Year', how='left')
            combined['Ads_Pct'] = combined['Gasto_Ads'] / combined['Total_Ventas']

            avg_ads_pct = combined['Ads_Pct'].mean()

            # Calcular ROAS promedio por plataforma
            roas_by_platform = self.df_ads.groupby('plataforma').agg({
                'venta_soles': 'sum',
                'gasto_soles': 'sum'
            }).reset_index()

            for _, row in roas_by_platform.iterrows():
                platform = row['plataforma']
                if row['gasto_soles'] > 0:
                    roas = row['venta_soles'] / row['gasto_soles']
                    avg_roas[platform] = roas

        # Aplicar porcentajes a la proyección 2026
        projected_costo_2026 = projected_sales_2026 * avg_costo_pct
        projected_ads_2026 = projected_sales_2026 * avg_ads_pct
        projected_margen_2026 = projected_sales_2026 * avg_margen_pct

        # Calcular utilidad neta (margen - gastos de ads)
        projected_utilidad_2026 = projected_margen_2026 - projected_ads_2026

        return {
            'year': 2026,
            'projected_sales': projected_sales_2026,
            'projected_costo': projected_costo_2026,
            'projected_ads': projected_ads_2026,
            'projected_margen': projected_margen_2026,
            'projected_utilidad': projected_utilidad_2026,
            'avg_costo_pct': avg_costo_pct * 100,
            'avg_margen_pct': avg_margen_pct * 100,
            'avg_ads_pct': avg_ads_pct * 100,
            'avg_roas': avg_roas,
            'growth_from_2025': sales_projection['avg_growth_pct']
        }

    def create_year_over_year_comparison(self, include_projection=True):
        """
        Crea gráfico comparativo año a año (2023 vs 2024 vs 2025)
        Muestra ventas totales y crecimiento porcentual

        Returns:
            JSON del gráfico o None
        """
        if self.df is None or self.df.empty:
            return None

        # Crear columna de año desde Fecha
        df_copy = self.df.copy()
        df_copy['Year'] = df_copy['Fecha'].dt.year

        # Agrupar ventas por año
        yearly_sales = df_copy.groupby('Year')['Precio Venta'].sum().reset_index()
        yearly_sales.columns = ['Año', 'Venta']
        yearly_sales = yearly_sales.sort_values('Año')

        if len(yearly_sales) < 2:
            return None

        # Calcular crecimiento año a año
        years = [int(y) for y in yearly_sales['Año'].tolist()]  # Convertir a enteros
        sales = yearly_sales['Venta'].tolist()

        # Calcular porcentaje de crecimiento
        growth_rates = []
        growth_labels = []

        for i in range(len(years)):
            if i == 0:
                growth_rates.append(None)  # No hay año anterior
                growth_labels.append("")
            else:
                prev_sales = sales[i-1]
                curr_sales = sales[i]
                growth_pct = ((curr_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
                growth_rates.append(growth_pct)
                growth_labels.append(f"+{growth_pct:.1f}%" if growth_pct >= 0 else f"{growth_pct:.1f}%")

        # Calcular proyección 2026 si se solicita
        projection_2026 = None
        projection_growth = None
        if include_projection:
            proj_data = self.calculate_2026_projection()
            if proj_data:
                projection_2026 = proj_data['combined']  # Usar método combinado
                # Calcular crecimiento proyectado
                last_year_sales = sales[-1]
                projection_growth = ((projection_2026 - last_year_sales) / last_year_sales * 100)

        fig = go.Figure()

        # Barras de ventas anuales (datos reales)
        fig.add_trace(go.Bar(
            x=years,
            y=sales,
            name='Ventas Reales',
            marker_color=['#95a5a6', '#3498db', '#2ecc71'],  # Gris, Azul, Verde
            text=[f"S/ {val:,.0f}" for val in sales],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Ventas: S/ %{y:,.2f}<extra></extra>'
        ))

        # Añadir barra de proyección 2026 si existe
        if projection_2026:
            fig.add_trace(go.Bar(
                x=[2026],
                y=[projection_2026],
                name='Proyección 2026',
                marker=dict(
                    color='rgba(155, 89, 182, 0.6)',  # Púrpura semi-transparente
                    pattern=dict(shape="/", solidity=0.3)  # Patrón de rayas
                ),
                text=[f"S/ {projection_2026:,.0f}"],
                textposition='outside',
                hovertemplate='<b>2026 (Proyección)</b><br>Ventas: S/ %{y:,.2f}<extra></extra>'
            ))

        # Añadir anotaciones de crecimiento
        annotations = []
        for i in range(1, len(years)):
            if growth_rates[i] is not None:
                annotations.append(
                    dict(
                        x=years[i],
                        y=sales[i],
                        text=growth_labels[i],
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor='#27ae60' if growth_rates[i] >= 0 else '#e74c3c',
                        ax=0,
                        ay=-40,
                        font=dict(size=14, color='#27ae60' if growth_rates[i] >= 0 else '#e74c3c', family='Arial Black')
                    )
                )

        # Añadir anotación para proyección 2026
        if projection_2026 and projection_growth is not None:
            growth_label_2026 = f"+{projection_growth:.1f}%" if projection_growth >= 0 else f"{projection_growth:.1f}%"
            annotations.append(
                dict(
                    x=2026,
                    y=projection_2026,
                    text=growth_label_2026 + " (proyectado)",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='#9b59b6',
                    ax=0,
                    ay=-40,
                    font=dict(size=14, color='#9b59b6', family='Arial Black')
                )
            )

        title_text = 'Comparación de Ventas Año a Año' + (' con Proyección 2026' if include_projection and projection_2026 else '')

        fig.update_layout(
            title=title_text,
            xaxis=dict(
                title='Año',
                tickmode='linear',
                tick0=min(years),
                dtick=1,
                range=[min(years)-0.5, 2026.5 if projection_2026 else max(years)+0.5]
            ),
            yaxis=dict(title='Ventas Totales (S/)', tickformat=',.0f'),
            template='plotly_white',
            showlegend=True if projection_2026 else False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if projection_2026 else {},
            annotations=annotations,
            height=450,
            bargap=0.2
        )

        return fig.to_json()

    def create_yoy_detailed_comparison(self):
        """
        Crea gráfico detallado de comparación año a año
        Incluye ventas, costos, utilidad y gastos de publicidad

        Returns:
            JSON del gráfico o None
        """
        if self.df is None or self.df.empty:
            return None

        # Crear columna de año desde Fecha
        df_copy = self.df.copy()
        df_copy['Year'] = df_copy['Fecha'].dt.year

        # Agrupar datos por año
        yearly_data = df_copy.groupby('Year').agg({
            'Precio Venta': 'sum',
            'Costo Producto': 'sum',
            'Margen': 'sum'
        }).reset_index()
        yearly_data.columns = ['Año', 'Venta', 'Costo', 'Utilidad']

        # Agregar datos de publicidad si están disponibles
        if self.df_ads is not None and not self.df_ads.empty:
            ads_by_year = self.df_ads.groupby('anio').agg({
                'gasto_soles': 'sum',
                'venta_soles': 'sum'
            }).reset_index()
            ads_by_year.columns = ['Año', 'Gasto_Ads', 'Venta_Ads']

            yearly_data = yearly_data.merge(ads_by_year, on='Año', how='left')
            yearly_data['Gasto_Ads'] = yearly_data['Gasto_Ads'].fillna(0)
            yearly_data['ROAS'] = yearly_data.apply(
                lambda row: row['Venta_Ads'] / row['Gasto_Ads'] if row['Gasto_Ads'] > 0 else 0,
                axis=1
            )
        else:
            yearly_data['Gasto_Ads'] = 0
            yearly_data['ROAS'] = 0

        yearly_data = yearly_data.sort_values('Año')

        if len(yearly_data) < 2:
            return None

        years = [int(y) for y in yearly_data['Año'].tolist()]  # Convertir a enteros

        fig = go.Figure()

        # Barras agrupadas
        metrics = [
            {'name': 'Ventas', 'column': 'Venta', 'color': '#3498db'},
            {'name': 'Costos', 'column': 'Costo', 'color': '#e74c3c'},
            {'name': 'Utilidad', 'column': 'Utilidad', 'color': '#2ecc71'},
            {'name': 'Gasto Ads', 'column': 'Gasto_Ads', 'color': '#f39c12'}
        ]

        for metric in metrics:
            values = yearly_data[metric['column']].tolist()
            fig.add_trace(go.Bar(
                name=metric['name'],
                x=years,
                y=values,
                marker_color=metric['color'],
                hovertemplate=f"<b>%{{x}}</b><br>{metric['name']}: S/ %{{y:,.2f}}<extra></extra>"
            ))

        fig.update_layout(
            title='Comparación Detallada Año a Año',
            xaxis=dict(
                title='Año',
                tickmode='linear',
                tick0=min(years),
                dtick=1
            ),
            yaxis=dict(title='Monto (S/)', tickformat=',.0f'),
            template='plotly_white',
            barmode='group',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )

        return fig.to_json()
