"""
Gráfico TOP 10 productos más vendidos del año 2025
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
print("GENERANDO GRAFICO TOP 10 PRODUCTOS MAS VENDIDOS 2025")
print("=" * 100)

# Query para obtener top 10 productos por ventas (monto)
query_ventas = """
SELECT
    producto,
    COUNT(*) as cantidad_transacciones,
    SUM(venta) as ventas_totales,
    SUM(venta - costo) as ganancia_total,
    ROUND(AVG(venta)::numeric, 2) as ticket_promedio
FROM ventas_master
WHERE EXTRACT(YEAR FROM fecha) = 2025
GROUP BY producto
ORDER BY ventas_totales DESC
LIMIT 10;
"""

df_ventas = pd.read_sql(query_ventas, engine)

# Query para obtener top 10 productos por cantidad de transacciones
query_cantidad = """
SELECT
    producto,
    COUNT(*) as cantidad_transacciones,
    SUM(venta) as ventas_totales,
    SUM(venta - costo) as ganancia_total
FROM ventas_master
WHERE EXTRACT(YEAR FROM fecha) = 2025
GROUP BY producto
ORDER BY cantidad_transacciones DESC
LIMIT 10;
"""

df_cantidad = pd.read_sql(query_cantidad, engine)
engine.dispose()

if df_ventas.empty:
    print("[ERROR] No hay datos para generar el grafico")
    exit(1)

print(f"\n[OK] TOP 10 por ventas obtenido")
print(f"[OK] TOP 10 por cantidad obtenido")

# Crear figura con 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle('TOP 10 Productos Mas Vendidos 2025',
             fontsize=16, fontweight='bold')

# Colores degradados
colors_ventas = plt.cm.Greens(np.linspace(0.4, 0.9, 10))
colors_cantidad = plt.cm.Blues(np.linspace(0.4, 0.9, 10))

# ============================================================================
# SUBPLOT 1: TOP 10 POR MONTO DE VENTAS
# ============================================================================
productos_ventas = df_ventas['producto'].tolist()
ventas = df_ventas['ventas_totales'].tolist()

# Acortar nombres de productos si son muy largos
productos_ventas_short = [p[:40] + '...' if len(p) > 40 else p for p in productos_ventas]

bars1 = ax1.barh(range(len(productos_ventas_short)), ventas, color=colors_ventas)

# Agregar valores en las barras
for i, (bar, val) in enumerate(zip(bars1, ventas)):
    ax1.text(val + max(ventas)*0.01, i, f'S/ {val:,.0f}',
             va='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('Ventas Totales (S/)', fontsize=12, fontweight='bold')
ax1.set_title('TOP 10 Productos por Monto de Ventas', fontsize=14, fontweight='bold', pad=15)
ax1.set_yticks(range(len(productos_ventas_short)))
ax1.set_yticklabels(productos_ventas_short, fontsize=10)
ax1.invert_yaxis()  # Mayor arriba
ax1.grid(axis='x', alpha=0.3)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'S/ {x/1000:.0f}k'))

# ============================================================================
# SUBPLOT 2: TOP 10 POR CANTIDAD DE TRANSACCIONES
# ============================================================================
productos_cantidad = df_cantidad['producto'].tolist()
cantidades = df_cantidad['cantidad_transacciones'].tolist()

productos_cantidad_short = [p[:40] + '...' if len(p) > 40 else p for p in productos_cantidad]

bars2 = ax2.barh(range(len(productos_cantidad_short)), cantidades, color=colors_cantidad)

# Agregar valores en las barras
for i, (bar, val) in enumerate(zip(bars2, cantidades)):
    ax2.text(val + max(cantidades)*0.01, i, f'{val:,}',
             va='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Cantidad de Transacciones', fontsize=12, fontweight='bold')
ax2.set_title('TOP 10 Productos por Cantidad Vendida', fontsize=14, fontweight='bold', pad=15)
ax2.set_yticks(range(len(productos_cantidad_short)))
ax2.set_yticklabels(productos_cantidad_short, fontsize=10)
ax2.invert_yaxis()  # Mayor arriba
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()

# Guardar gráfico
output_path = 'scripts/top10_productos_2025.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] Grafico guardado en: {output_path}")
plt.close()

# ============================================================================
# ESTADISTICAS DETALLADAS
# ============================================================================
print("\n" + "=" * 100)
print("TOP 10 PRODUCTOS POR VENTAS (MONTO)")
print("=" * 100)
print(df_ventas.to_string(index=False))

ventas_top10 = df_ventas['ventas_totales'].sum()
ganancia_top10 = df_ventas['ganancia_total'].sum()

print(f"\nTOTAL TOP 10:")
print(f"  Ventas:    S/ {ventas_top10:,.2f}")
print(f"  Ganancia:  S/ {ganancia_top10:,.2f}")
print(f"  Margen:    {(ganancia_top10/ventas_top10*100):.1f}%")

# Calcular porcentaje del total
query_total = """
SELECT
    SUM(venta) as ventas_totales,
    SUM(venta - costo) as ganancia_total
FROM ventas_master
WHERE EXTRACT(YEAR FROM fecha) = 2025;
"""

engine = create_engine(get_db_url())
df_total = pd.read_sql(query_total, engine)
engine.dispose()

ventas_totales_2025 = df_total['ventas_totales'].iloc[0]
ganancia_total_2025 = df_total['ganancia_total'].iloc[0]

pct_ventas = (ventas_top10 / ventas_totales_2025 * 100) if ventas_totales_2025 > 0 else 0
pct_ganancia = (ganancia_top10 / ganancia_total_2025 * 100) if ganancia_total_2025 > 0 else 0

print(f"\nPORCENTAJE DEL TOTAL 2025:")
print(f"  Ventas:    {pct_ventas:.1f}%")
print(f"  Ganancia:  {pct_ganancia:.1f}%")

print("\n" + "=" * 100)
print("TOP 10 PRODUCTOS POR CANTIDAD")
print("=" * 100)
print(df_cantidad.to_string(index=False))

print("\n" + "=" * 100)
