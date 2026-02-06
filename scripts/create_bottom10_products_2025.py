"""
Gráfico TOP 10 productos HUESO (menos vendidos) del año 2025
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
print("GENERANDO GRAFICO TOP 10 PRODUCTOS HUESO (MENOS VENDIDOS) 2025")
print("=" * 100)

# Query para obtener bottom 10 productos por ventas (monto)
# Excluyendo productos con muy pocas transacciones (al menos 2)
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
HAVING COUNT(*) >= 2
ORDER BY ventas_totales ASC
LIMIT 10;
"""

df_ventas = pd.read_sql(query_ventas, engine)

# Query para obtener bottom 10 productos por cantidad de transacciones
query_cantidad = """
SELECT
    producto,
    COUNT(*) as cantidad_transacciones,
    SUM(venta) as ventas_totales,
    SUM(venta - costo) as ganancia_total,
    ROUND((SUM(venta - costo) / SUM(venta) * 100)::numeric, 1) as margen_pct
FROM ventas_master
WHERE EXTRACT(YEAR FROM fecha) = 2025
GROUP BY producto
HAVING COUNT(*) >= 2
ORDER BY cantidad_transacciones ASC, ventas_totales ASC
LIMIT 10;
"""

df_cantidad = pd.read_sql(query_cantidad, engine)
engine.dispose()

if df_ventas.empty:
    print("[ERROR] No hay datos para generar el grafico")
    exit(1)

print(f"\n[OK] BOTTOM 10 por ventas obtenido")
print(f"[OK] BOTTOM 10 por cantidad obtenido")

# Crear figura con 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
fig.suptitle('TOP 10 Productos HUESO (Menor Rotacion) 2025',
             fontsize=16, fontweight='bold')

# Colores degradados (rojos para indicar problema)
colors_ventas = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
colors_cantidad = plt.cm.Oranges(np.linspace(0.4, 0.9, 10))

# ============================================================================
# SUBPLOT 1: BOTTOM 10 POR MONTO DE VENTAS
# ============================================================================
productos_ventas = df_ventas['producto'].tolist()
ventas = df_ventas['ventas_totales'].tolist()

# Acortar nombres de productos si son muy largos
productos_ventas_short = [p[:40] + '...' if len(p) > 40 else p for p in productos_ventas]

bars1 = ax1.barh(range(len(productos_ventas_short)), ventas, color=colors_ventas)

# Agregar valores en las barras
for i, (bar, val) in enumerate(zip(bars1, ventas)):
    ax1.text(val + max(ventas)*0.02, i, f'S/ {val:,.0f}',
             va='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('Ventas Totales (S/)', fontsize=12, fontweight='bold')
ax1.set_title('BOTTOM 10 Productos por Menor Monto de Ventas',
              fontsize=14, fontweight='bold', pad=15, color='#c0392b')
ax1.set_yticks(range(len(productos_ventas_short)))
ax1.set_yticklabels(productos_ventas_short, fontsize=10)
ax1.invert_yaxis()  # Menor arriba
ax1.grid(axis='x', alpha=0.3)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'S/ {x:,.0f}'))

# Agregar texto de advertencia
ax1.text(0.98, 0.02, 'Productos de baja rotacion - Revisar inventario',
         transform=ax1.transAxes, fontsize=10, ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# SUBPLOT 2: BOTTOM 10 POR CANTIDAD DE TRANSACCIONES
# ============================================================================
productos_cantidad = df_cantidad['producto'].tolist()
cantidades = df_cantidad['cantidad_transacciones'].tolist()

productos_cantidad_short = [p[:40] + '...' if len(p) > 40 else p for p in productos_cantidad]

bars2 = ax2.barh(range(len(productos_cantidad_short)), cantidades, color=colors_cantidad)

# Agregar valores en las barras
for i, (bar, val) in enumerate(zip(bars2, cantidades)):
    ax2.text(val + max(cantidades)*0.02, i, f'{val:,}',
             va='center', fontsize=9, fontweight='bold')

ax2.set_xlabel('Cantidad de Transacciones', fontsize=12, fontweight='bold')
ax2.set_title('BOTTOM 10 Productos por Menor Cantidad Vendida',
              fontsize=14, fontweight='bold', pad=15, color='#d35400')
ax2.set_yticks(range(len(productos_cantidad_short)))
ax2.set_yticklabels(productos_cantidad_short, fontsize=10)
ax2.invert_yaxis()  # Menor arriba
ax2.grid(axis='x', alpha=0.3)

# Agregar texto de advertencia
ax2.text(0.98, 0.02, 'Considerar: descontinuar, liquidar o promocionar',
         transform=ax2.transAxes, fontsize=10, ha='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Guardar gráfico
output_path = 'scripts/bottom10_productos_2025.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[OK] Grafico guardado en: {output_path}")
plt.close()

# ============================================================================
# ESTADISTICAS DETALLADAS
# ============================================================================
print("\n" + "=" * 100)
print("BOTTOM 10 PRODUCTOS POR VENTAS (MENOR MONTO)")
print("=" * 100)
print(df_ventas.to_string(index=False))

ventas_bottom10 = df_ventas['ventas_totales'].sum()
ganancia_bottom10 = df_ventas['ganancia_total'].sum()

print(f"\nTOTAL BOTTOM 10:")
print(f"  Ventas:    S/ {ventas_bottom10:,.2f}")
print(f"  Ganancia:  S/ {ganancia_bottom10:,.2f}")
if ventas_bottom10 > 0:
    print(f"  Margen:    {(ganancia_bottom10/ventas_bottom10*100):.1f}%")

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

pct_ventas = (ventas_bottom10 / ventas_totales_2025 * 100) if ventas_totales_2025 > 0 else 0
pct_ganancia = (ganancia_bottom10 / ganancia_total_2025 * 100) if ganancia_total_2025 > 0 else 0

print(f"\nPORCENTAJE DEL TOTAL 2025:")
print(f"  Ventas:    {pct_ventas:.2f}%")
print(f"  Ganancia:  {pct_ganancia:.2f}%")

print("\n" + "=" * 100)
print("BOTTOM 10 PRODUCTOS POR CANTIDAD")
print("=" * 100)
print(df_cantidad.to_string(index=False))

print("\n" + "=" * 100)
print("RECOMENDACIONES")
print("=" * 100)
print("1. LIQUIDAR: Productos con margenes bajos y poca rotacion")
print("2. PROMOCIONAR: Productos con margenes altos pero poca rotacion")
print("3. DESCONTINUAR: Productos que no se venden en 3+ meses")
print("4. ANALIZAR PRECIOS: Pueden estar muy caros vs competencia")
print("5. REVISAR STOCK: No sobre-inventariar estos productos")
print("=" * 100)
