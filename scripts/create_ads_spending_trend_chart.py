"""
Gráfico histórico mensual de gastos en publicidad por plataforma y año
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
print("GENERANDO GRAFICO HISTORICO DE GASTOS EN PUBLICIDAD")
print("=" * 100)

# Query para obtener gastos en publicidad por mes, año y plataforma
query = """
SELECT
    anio,
    mes,
    plataforma,
    SUM(gasto_soles) as gasto_total
FROM ads_costs
WHERE anio IN (2023, 2024, 2025)
GROUP BY anio, mes, plataforma
ORDER BY anio, mes, plataforma;
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

# Crear figura con 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14))
fig.suptitle('Historico Mensual de Gastos en Publicidad por Plataforma',
             fontsize=16, fontweight='bold')

# Colores por año
colores_anos = {
    2023: '#3498db',  # Azul
    2024: '#e67e22',  # Naranja
    2025: '#2ecc71'   # Verde
}

# Colores por plataforma
colores_plataformas = {
    'META': '#4267B2',      # Azul Facebook
    'TIKTOK': '#EE1D52'     # Rosa TikTok
}

# ============================================================================
# SUBPLOT 1: GASTO TOTAL (META + TIKTOK)
# ============================================================================
df_total = df.groupby(['anio', 'mes'])['gasto_total'].sum().reset_index()

for anio in sorted(df_total['anio'].unique()):
    df_anio = df_total[df_total['anio'] == anio].copy()
    df_anio = df_anio.sort_values('mes')

    meses = df_anio['mes'].tolist()
    gastos = df_anio['gasto_total'].tolist()

    # Crear lista de meses completa (1-12) con None para meses sin datos
    gastos_completos = [None] * 12
    for i, mes in enumerate(meses):
        gastos_completos[int(mes)-1] = gastos[i]

    ax1.plot(range(1, 13), gastos_completos,
             marker='o', linewidth=2.5, markersize=8,
             color=colores_anos.get(anio, '#95a5a6'),
             label=f'{int(anio)}')

ax1.set_xlabel('Mes', fontsize=12, fontweight='bold')
ax1.set_ylabel('Gasto Total (S/)', fontsize=12, fontweight='bold')
ax1.set_title('Gasto Total en Publicidad (META + TIKTOK)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(meses_nombres)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'S/ {x:,.0f}'))

# ============================================================================
# SUBPLOT 2: GASTO META
# ============================================================================
df_meta = df[df['plataforma'] == 'META'].copy()

for anio in sorted(df_meta['anio'].unique()):
    df_anio = df_meta[df_meta['anio'] == anio].copy()
    df_anio = df_anio.sort_values('mes')

    meses = df_anio['mes'].tolist()
    gastos = df_anio['gasto_total'].tolist()

    gastos_completos = [None] * 12
    for i, mes in enumerate(meses):
        gastos_completos[int(mes)-1] = gastos[i]

    ax2.plot(range(1, 13), gastos_completos,
             marker='s', linewidth=2.5, markersize=8,
             color=colores_anos.get(anio, '#95a5a6'),
             label=f'{int(anio)}')

ax2.set_xlabel('Mes', fontsize=12, fontweight='bold')
ax2.set_ylabel('Gasto META (S/)', fontsize=12, fontweight='bold')
ax2.set_title('Gasto en META (Facebook/Instagram)', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(meses_nombres)
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(axis='y', alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'S/ {x:,.0f}'))

# ============================================================================
# SUBPLOT 3: GASTO TIKTOK
# ============================================================================
df_tiktok = df[df['plataforma'] == 'TIKTOK'].copy()

for anio in sorted(df_tiktok['anio'].unique()):
    df_anio = df_tiktok[df_tiktok['anio'] == anio].copy()
    df_anio = df_anio.sort_values('mes')

    meses = df_anio['mes'].tolist()
    gastos = df_anio['gasto_total'].tolist()

    gastos_completos = [None] * 12
    for i, mes in enumerate(meses):
        gastos_completos[int(mes)-1] = gastos[i]

    ax3.plot(range(1, 13), gastos_completos,
             marker='^', linewidth=2.5, markersize=8,
             color=colores_anos.get(anio, '#95a5a6'),
             label=f'{int(anio)}')

ax3.set_xlabel('Mes', fontsize=12, fontweight='bold')
ax3.set_ylabel('Gasto TIKTOK (S/)', fontsize=12, fontweight='bold')
ax3.set_title('Gasto en TIKTOK', fontsize=14, fontweight='bold', pad=15)
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(meses_nombres)
ax3.legend(loc='upper left', fontsize=11)
ax3.grid(axis='y', alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'S/ {x:,.0f}'))

plt.tight_layout()

# Guardar gráfico
output_path = 'scripts/historico_gastos_publicidad.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] Grafico guardado en: {output_path}")
plt.close()

# ============================================================================
# ESTADISTICAS DE TENDENCIA
# ============================================================================
print("\n" + "=" * 100)
print("ANALISIS DE GASTOS EN PUBLICIDAD ANO A ANO")
print("=" * 100)

# Totales por año
df_anual = df.groupby('anio')['gasto_total'].sum().reset_index()
df_anual = df_anual.sort_values('anio')

for _, row in df_anual.iterrows():
    anio = int(row['anio'])
    gasto_total = row['gasto_total']

    # Desglose por plataforma
    gasto_meta = df[(df['anio'] == anio) & (df['plataforma'] == 'META')]['gasto_total'].sum()
    gasto_tiktok = df[(df['anio'] == anio) & (df['plataforma'] == 'TIKTOK')]['gasto_total'].sum()

    print(f"\n{anio}:")
    print(f"  Gasto Total:     S/ {gasto_total:,.2f}")
    print(f"    META:          S/ {gasto_meta:,.2f} ({gasto_meta/gasto_total*100:.1f}%)")
    print(f"    TIKTOK:        S/ {gasto_tiktok:,.2f} ({gasto_tiktok/gasto_total*100:.1f}%)")

# Calcular crecimiento año a año
print("\n" + "=" * 100)
print("CRECIMIENTO EN GASTOS ANO A ANO")
print("=" * 100)

anos = sorted(df_anual['anio'].unique())
for i in range(1, len(anos)):
    anio_anterior = anos[i-1]
    anio_actual = anos[i]

    gasto_anterior = df_anual[df_anual['anio'] == anio_anterior]['gasto_total'].iloc[0]
    gasto_actual = df_anual[df_anual['anio'] == anio_actual]['gasto_total'].iloc[0]

    crecimiento = ((gasto_actual - gasto_anterior) / gasto_anterior * 100) if gasto_anterior > 0 else 0

    print(f"\n{int(anio_anterior)} -> {int(anio_actual)}:")
    print(f"  Gasto Total: {crecimiento:+.1f}% (S/ {gasto_actual - gasto_anterior:+,.2f})")

    # Desglose por plataforma
    meta_anterior = df[(df['anio'] == anio_anterior) & (df['plataforma'] == 'META')]['gasto_total'].sum()
    meta_actual = df[(df['anio'] == anio_actual) & (df['plataforma'] == 'META')]['gasto_total'].sum()
    crecimiento_meta = ((meta_actual - meta_anterior) / meta_anterior * 100) if meta_anterior > 0 else 0

    tiktok_anterior = df[(df['anio'] == anio_anterior) & (df['plataforma'] == 'TIKTOK')]['gasto_total'].sum()
    tiktok_actual = df[(df['anio'] == anio_actual) & (df['plataforma'] == 'TIKTOK')]['gasto_total'].sum()
    crecimiento_tiktok = ((tiktok_actual - tiktok_anterior) / tiktok_anterior * 100) if tiktok_anterior > 0 else 0

    print(f"    META:   {crecimiento_meta:+.1f}% (S/ {meta_actual - meta_anterior:+,.2f})")
    print(f"    TIKTOK: {crecimiento_tiktok:+.1f}% (S/ {tiktok_actual - tiktok_anterior:+,.2f})")

print("\n" + "=" * 100)
