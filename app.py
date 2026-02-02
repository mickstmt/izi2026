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

        stats = processor.get_historical_metrics(base_year)
        proj_df = processor.run_projection(growth, roas_meta, roas_tiktok, base_year)

        # Final safety check on numbers
        print(f"Projected Annual Sales: {proj_df['Ventas'].sum():,.2f}")

        trend_chart = processor.create_trend_chart(proj_df)
        # Usar gráfico de composición basado en tendencias históricas
        comp_chart = processor.create_trend_based_composition_chart()
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
                               comp_chart=comp_chart,
                               top_products_chart=top_products_chart,
                               yearly_chart=yearly_chart,
                               category_chart=category_chart,
                               historical_monthly_chart=historical_monthly_chart,
                               # Nuevos parámetros de publicidad
                               ads_spending_chart=ads_spending_chart,
                               roas_comparison_chart=roas_comparison_chart,
                               sales_goal_chart=sales_goal_chart,
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

    proj_df = processor.run_projection(growth, roas_meta, roas_tiktok, base_year)

    trend_chart = processor.create_trend_chart(proj_df)
    # Usar gráfico de composición basado en tendencias históricas
    comp_chart = processor.create_trend_based_composition_chart()

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
        'comp_chart': comp_chart,
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

if __name__ == '__main__':
    with app.app_context():
        processor.load_data()
    app.run(debug=True, port=5000)
