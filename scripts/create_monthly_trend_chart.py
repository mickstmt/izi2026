"""
Gráfico histórico mensual de ventas y ganancias por año con tendencias
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine
from db_config import get_db_url

engine = create_engine(get_db_url())

print("=" * 100)
print("GENERANDO GRAFICO HISTORICO MENSUAL DE VENTAS Y GANANCIAS")
print("=" * 100)

# Query para obtener ventas y ganancias por mes y año
query = """
SELECT
    EXTRACT(YEAR FROM fecha) as anio,
    EXTRACT(MONTH FROM fecha) as mes,
    SUM(venta) as ventas,
    SUM(venta - costo) as ganancia
FROM ventas_master
WHERE EXTRACT(YEAR FROM fecha) IN (2023, 2024, 2025)
GROUP BY EXTRACT(YEAR FROM fecha), EXTRACT(MONTH FROM fecha)
ORDER BY anio, mes;
"""

df = pd.read_sql(query, engine)
engine.dispose()

if df.empty:
    print("[ERROR] No hay datos para generar el grafico")
    exit(1)

print(f"\n[OK] Datos obtenidos: {len(df)} registros")

# Nombres de meses
meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

# Crear figura con 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle('Historico Mensual de Ventas y Ganancias por Ano',
             fontsize=16, fontweight='bold')

# Colores por año
colores = {
    2023: '#3498db',  # Azul
    2024: '#e67e22',  # Naranja
    2025: '#2ecc71'   # Verde
}

# ============================================================================
# SUBPLOT 1: VENTAS
# ============================================================================
for anio in sorted(df['anio'].unique()):
    df_anio = df[df['anio'] == anio].copy()
    df_anio = df_anio.sort_values('mes')

    meses = df_anio['mes'].tolist()
    ventas = df_anio['ventas'].tolist()

    # Crear lista de meses completa (1-12) con NaN para meses sin datos
    ventas_completas = [None] * 12
    for i, mes in enumerate(meses):
        ventas_completas[int(mes)-1] = ventas[i]

    ax1.plot(range(1, 13), ventas_completas,
             marker='o', linewidth=2, markersize=8,
             color=colores.get(anio, '#95a5a6'),
             label=f'{int(anio)}')

ax1.set_xlabel('Mes', fontsize=12, fontweight='bold')
ax1.set_ylabel('Ventas (S/)', fontsize=12, fontweight='bold')
ax1.set_title('Ventas Mensuales por Ano', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(meses_nombres)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'S/ {x/1000:.0f}k'))

# ============================================================================
# SUBPLOT 2: GANANCIAS
# ============================================================================
for anio in sorted(df['anio'].unique()):
    df_anio = df[df['anio'] == anio].copy()
    df_anio = df_anio.sort_values('mes')

    meses = df_anio['mes'].tolist()
    ganancias = df_anio['ganancia'].tolist()

    # Crear lista de meses completa (1-12) con NaN para meses sin datos
    ganancias_completas = [None] * 12
    for i, mes in enumerate(meses):
        ganancias_completas[int(mes)-1] = ganancias[i]

    ax2.plot(range(1, 13), ganancias_completas,
             marker='s', linewidth=2, markersize=8,
             color=colores.get(anio, '#95a5a6'),
             label=f'{int(anio)}')

ax2.set_xlabel('Mes', fontsize=12, fontweight='bold')
ax2.set_ylabel('Ganancia (S/)', fontsize=12, fontweight='bold')
ax2.set_title('Ganancias Mensuales por Ano', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(meses_nombres)
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(axis='y', alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'S/ {x/1000:.0f}k'))

plt.tight_layout()

# Guardar gráfico
output_path = 'scripts/historico_mensual_ventas_ganancias.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] Grafico guardado en: {output_path}")
plt.close()

# ============================================================================
# ESTADISTICAS DE TENDENCIA
# ============================================================================
print("\n" + "=" * 100)
print("ANALISIS DE TENDENCIAS ANO A ANO")
print("=" * 100)

for anio in sorted(df['anio'].unique()):
    df_anio = df[df['anio'] == anio]
    ventas_total = df_anio['ventas'].sum()
    ganancia_total = df_anio['ganancia'].sum()
    ventas_promedio = df_anio['ventas'].mean()
    ganancia_promedio = df_anio['ganancia'].mean()

    print(f"\n{int(anio)}:")
    print(f"  Ventas Totales:      S/ {ventas_total:,.2f}")
    print(f"  Ganancias Totales:   S/ {ganancia_total:,.2f}")
    print(f"  Ventas Promedio/Mes: S/ {ventas_promedio:,.2f}")
    print(f"  Ganancia Promedio/Mes: S/ {ganancia_promedio:,.2f}")

# Calcular crecimiento año a año
print("\n" + "=" * 100)
print("CRECIMIENTO ANO A ANO")
print("=" * 100)

anos = sorted(df['anio'].unique())
for i in range(1, len(anos)):
    anio_anterior = anos[i-1]
    anio_actual = anos[i]

    ventas_anterior = df[df['anio'] == anio_anterior]['ventas'].sum()
    ventas_actual = df[df['anio'] == anio_actual]['ventas'].sum()

    ganancia_anterior = df[df['anio'] == anio_anterior]['ganancia'].sum()
    ganancia_actual = df[df['anio'] == anio_actual]['ganancia'].sum()

    crecimiento_ventas = ((ventas_actual - ventas_anterior) / ventas_anterior * 100) if ventas_anterior > 0 else 0
    crecimiento_ganancia = ((ganancia_actual - ganancia_anterior) / ganancia_anterior * 100) if ganancia_anterior > 0 else 0

    print(f"\n{int(anio_anterior)} -> {int(anio_actual)}:")
    print(f"  Ventas:    {crecimiento_ventas:+.1f}% (S/ {ventas_actual - ventas_anterior:+,.2f})")
    print(f"  Ganancias: {crecimiento_ganancia:+.1f}% (S/ {ganancia_actual - ganancia_anterior:+,.2f})")

print("\n" + "=" * 100)
