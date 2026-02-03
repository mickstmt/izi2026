from flask import Flask, render_template, request, jsonify
from engine.data_processor import DataProcessor
import os

app = Flask(__name__)

# Initialize DataProcessor
# You can put your Master Excel in the 'data' folder
DATA_PATH = os.path.join(os.getcwd(), 'data', 'Master_Excel_2025.xlsx')
processor = DataProcessor(DATA_PATH)

@app.route('/')
def index():
    try:
        if processor.df is None:
            processor.load_data()

        # Diagnostic: check how many years we have
        available_years = processor.df['Fecha'].dt.year.unique()
        print(f"Available years in index: {available_years}")

        # Default values for projection
        growth = 10
        roas_meta = 12.0
        roas_tiktok = 8.0
        base_year = 2025
        projection_type = 'sales'  # Default: crecimiento en ventas

        stats = processor.get_historical_metrics(base_year)
        proj_df = processor.run_projection(growth, roas_meta, roas_tiktok, base_year, projection_type)

        # Final safety check on numbers
        print(f"Projected Annual Sales: {proj_df['Ventas'].sum():,.2f}")

        trend_chart = processor.create_trend_chart(proj_df)
        # Nuevos gráficos de análisis de productos
        top_20_chart = processor.create_top_20_products_chart()
        dead_stock_chart = processor.create_dead_stock_chart(months_threshold=6)
        top_products_chart = processor.get_top_products()
        yearly_chart = processor.get_yearly_history_chart()
        category_chart = processor.get_category_analysis()
        historical_monthly_chart = processor.get_historical_monthly_trends()

        # ============================================================
        # NUEVOS GRÁFICOS Y MÉTRICAS DE PUBLICIDAD
        # ============================================================

        # Cargar datos de ads si no están cargados
        if processor.df_ads is None:
            processor.load_ads_data()

        # Gráficos de publicidad
        ads_spending_chart = None
        roas_comparison_chart = None
        sales_goal_chart = None
        attribution_chart = None
        ads_metrics = None
        ads_attribution = None
        monthly_roas_chart = None
        monthly_spending_chart = None
        monthly_projection_chart = None

        if processor.df_ads is not None and not processor.df_ads.empty:
            # Gráfico de inversión en ads
            ads_spending_chart = processor.create_ads_spending_chart()

            # Gráfico ROAS Real vs Esperado
            roas_comparison_chart = processor.create_roas_comparison_chart(year=2025, roas_meta=roas_meta, roas_tiktok=roas_tiktok)

            # Gráfico Venta Real vs Meta
            sales_goal_chart = processor.create_sales_goal_comparison_chart(year=2025)

            # Gráfico de distribución Ads vs Orgánicas
            attribution_chart = processor.create_sales_attribution_chart(year=2025)

            # Métricas de ads
            ads_metrics = processor.calculate_ads_metrics(year=2025)

            # Análisis de atribución (% ventas de publicidad vs orgánicas)
            ads_attribution = processor.get_ads_attribution_analysis(year=2025)

            # Nuevos gráficos mensuales
            monthly_roas_chart = processor.create_monthly_roas_trend(year=2025)
            monthly_spending_chart = processor.create_monthly_ads_spending(year=2025)
            monthly_projection_chart = processor.create_monthly_investment_projection(growth, roas_meta, roas_tiktok, base_year)

        # Gráficos de comparación año a año
        yoy_comparison_chart = processor.create_year_over_year_comparison()
        yoy_detailed_chart = processor.create_yoy_detailed_comparison()

        summary = {
            'total_sales': proj_df['Ventas'].sum(),
            'total_profit': proj_df['Utilidad Neta'].sum(),
            'total_cogs': proj_df['Costos Mercaderia'].sum(),
            'total_ads': proj_df['Gasto Ads'].sum()
        }

        return render_template('dashboard.html',
                               summary=summary,
                               growth=growth,
                               roas_meta=roas_meta,
                               roas_tiktok=roas_tiktok,
                               trend_chart=trend_chart,
                               top_20_chart=top_20_chart,
                               dead_stock_chart=dead_stock_chart,
                               top_products_chart=top_products_chart,
                               yearly_chart=yearly_chart,
                               category_chart=category_chart,
                               historical_monthly_chart=historical_monthly_chart,
                               # Nuevos parámetros de publicidad
                               ads_spending_chart=ads_spending_chart,
                               roas_comparison_chart=roas_comparison_chart,
                               sales_goal_chart=sales_goal_chart,
                               attribution_chart=attribution_chart,
                               ads_metrics=ads_metrics,
                               ads_attribution=ads_attribution,
                               # Gráficos mensuales
                               monthly_roas_chart=monthly_roas_chart,
                               monthly_spending_chart=monthly_spending_chart,
                               monthly_projection_chart=monthly_projection_chart,
                               # Comparación año a año
                               yoy_comparison_chart=yoy_comparison_chart,
                               yoy_detailed_chart=yoy_detailed_chart)
    except Exception as e:
        return f"<h1>Error en la Aplicación</h1><p>{str(e)}</p><p>Por favor revisa que las columnas del Excel sean correctas.</p>"

@app.route('/recalculate', methods=['POST'])
def recalculate():
    data = request.json
    growth = float(data.get('growth', 10))
    roas_meta = max(0.1, float(data.get('roas_meta', 12.0)))
    roas_tiktok = max(0.1, float(data.get('roas_tiktok', 8.0)))
    base_year = data.get('base_year', '2025')
    projection_type = data.get('projection_type', 'sales')  # 'sales' o 'profit'

    print(f"=== DEBUG RECALCULATE: projection_type={projection_type}, growth={growth}")

    proj_df = processor.run_projection(growth, roas_meta, roas_tiktok, base_year, projection_type)

    trend_chart = processor.create_trend_chart(proj_df)

    # Regenerar gráfico ROAS con valores configurados
    roas_comparison_chart = processor.create_roas_comparison_chart(year=2025, roas_meta=roas_meta, roas_tiktok=roas_tiktok)

    # Regenerar gráfico de proyección mensual con nuevos ROAS
    monthly_projection_chart = processor.create_monthly_investment_projection(growth, roas_meta, roas_tiktok, base_year)

    summary = {
        'total_sales': f"S/ {proj_df['Ventas'].sum():,.2f}",
        'total_profit': f"S/ {proj_df['Utilidad Neta'].sum():,.2f}",
        'total_cogs': f"S/ {proj_df['Costos Mercaderia'].sum():,.2f}",
        'total_ads': f"S/ {proj_df['Gasto Ads'].sum():,.2f}"
    }

    return jsonify({
        'summary': summary,
        'trend_chart': trend_chart,
        'roas_comparison_chart': roas_comparison_chart,
        'monthly_projection_chart': monthly_projection_chart
    })

# Endpoints API para gráficos individuales con selector de año
@app.route('/api/roas_comparison/<int:year>', methods=['GET'])
def api_roas_comparison(year):
    """Retorna gráfico ROAS Real vs Esperado para el año especificado"""
    # Obtener ROAS targets de los parámetros query o usar defaults
    roas_meta = float(request.args.get('roas_meta', 12.0))
    roas_tiktok = float(request.args.get('roas_tiktok', 8.0))

    chart = processor.create_roas_comparison_chart(year=year, roas_meta=roas_meta, roas_tiktok=roas_tiktok)
    return jsonify({'chart': chart})

@app.route('/api/sales_goal/<int:year>', methods=['GET'])
def api_sales_goal(year):
    """Retorna gráfico Venta Real vs Meta para el año especificado"""
    chart = processor.create_sales_goal_comparison_chart(year=year)
    return jsonify({'chart': chart})

@app.route('/api/sales_attribution/<int:year>', methods=['GET'])
def api_sales_attribution(year):
    """Retorna gráfico de distribución Ventas Ads vs Orgánicas para el año especificado"""
    chart = processor.create_sales_attribution_chart(year=year)
    return jsonify({'chart': chart})

@app.route('/api/monthly_roas/<int:year>', methods=['GET'])
def api_monthly_roas(year):
    """Retorna gráfico ROAS Histórico Mes a Mes para el año especificado"""
    chart = processor.create_monthly_roas_trend(year=year)
    return jsonify({'chart': chart})

@app.route('/api/monthly_spending/<int:year>', methods=['GET'])
def api_monthly_spending(year):
    """Retorna gráfico Inversión en Publicidad Mes a Mes para el año especificado"""
    chart = processor.create_monthly_ads_spending(year=year)
    return jsonify({'chart': chart})

@app.route('/api/historical_budgets/<int:year>', methods=['GET'])
def api_historical_budgets(year):
    """Retorna gastos históricos reales por mes y plataforma para un año específico"""
    try:
        if processor.df_ads is None or processor.df_ads.empty:
            return jsonify({'error': 'No hay datos históricos de publicidad disponibles'}), 404

        # Filtrar por año
        df_year = processor.df_ads[processor.df_ads['anio'] == year]

        if df_year.empty:
            return jsonify({'error': f'No hay datos para el año {year}'}), 404

        # Mapeo de meses
        meses_esp = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
            'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
            'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }

        # Crear diccionario de resultado
        budgets = {
            'META': {},
            'TIKTOK': {}
        }

        # Agrupar por plataforma y mes
        for _, row in df_year.iterrows():
            platform = row['plataforma']
            mes_nombre = row['mes_nombre'].lower()
            mes_num = meses_esp.get(mes_nombre)
            gasto = float(row['gasto_soles'])

            if mes_num and platform in budgets:
                # Si ya existe, sumar (en caso de múltiples registros)
                if mes_num in budgets[platform]:
                    budgets[platform][mes_num] += gasto
                else:
                    budgets[platform][mes_num] = gasto

        # Llenar meses faltantes con 0
        for platform in ['META', 'TIKTOK']:
            for mes in range(1, 13):
                if mes not in budgets[platform]:
                    budgets[platform][mes] = 0

        return jsonify({
            'year': year,
            'budgets': budgets
        })

    except Exception as e:
        print(f"Error en historical_budgets: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

@app.route('/api/custom_budget_projection', methods=['POST'])
def api_custom_budget_projection():
    """Calcula proyección basada en presupuestos mensuales personalizados"""
    try:
        data = request.json
        base_year = data.get('base_year', 2025)
        monthly_budgets = data.get('monthly_budgets')

        # Validación básica
        if not monthly_budgets or 'META' not in monthly_budgets or 'TIKTOK' not in monthly_budgets:
            return jsonify({'error': 'Estructura de presupuesto inválida. Debe contener claves META y TIKTOK'}), 400

        # Calcular proyección custom
        custom_df = processor.run_custom_budget_projection(monthly_budgets, base_year)

        # Calcular proyección ROAS para comparación (usar valores por defecto)
        growth = 10
        roas_meta = 12.0
        roas_tiktok = 8.0
        roas_df = processor.run_projection(growth, roas_meta, roas_tiktok, base_year)

        # Generar gráfico comparativo
        comparison_chart = processor.create_custom_vs_roas_comparison_chart(custom_df, roas_df)

        # Análisis de eficiencia
        historical_roas = processor.get_historical_roas_by_month(base_year)
        efficiency = processor.analyze_budget_efficiency(custom_df, monthly_budgets, historical_roas)

        # Calcular summary con totales
        summary = {
            'custom_total_sales': float(custom_df['Ventas_Totales'].sum()),
            'custom_total_profit': float(custom_df['Utilidad_Neta'].sum()),
            'custom_total_budget': float(custom_df['Gasto_Ads'].sum()),
            'roas_total_sales': float(roas_df['Ventas'].sum()),
            'roas_total_profit': float(roas_df['Utilidad Neta'].sum()),
            'roas_total_budget': float(roas_df['Gasto Ads'].sum())
        }

        # Calcular diferencias
        summary['sales_difference'] = summary['custom_total_sales'] - summary['roas_total_sales']
        summary['sales_difference_pct'] = (summary['sales_difference'] / summary['roas_total_sales'] * 100) if summary['roas_total_sales'] > 0 else 0
        summary['profit_difference'] = summary['custom_total_profit'] - summary['roas_total_profit']
        summary['profit_difference_pct'] = (summary['profit_difference'] / summary['roas_total_profit'] * 100) if summary['roas_total_profit'] > 0 else 0

        return jsonify({
            'custom_projection': custom_df.to_dict('records'),
            'roas_projection': roas_df.to_dict('records'),
            'comparison_chart': comparison_chart,
            'efficiency_analysis': efficiency,
            'summary': summary,
            'historical_roas': historical_roas
        })

    except ValueError as e:
        # Errores de validación (estructura incorrecta, valores negativos, etc.)
        return jsonify({'error': str(e)}), 422
    except Exception as e:
        # Errores inesperados
        print(f"Error en custom_budget_projection: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

if __name__ == '__main__':
    with app.app_context():
        processor.load_data()
    app.run(debug=True, port=5000)
