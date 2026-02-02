import pandas as pd
from sqlalchemy import create_engine, text
import os
import sys

# Agregar el directorio raíz al path para importar db_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_config import get_db_url

def sync_data():
    engine = create_engine(get_db_url())
    
    # Rutas de archivos
    data_dir = os.path.join(os.getcwd(), 'data')
    pos_file = os.path.join(data_dir, 'Master_Excel_2025.xlsx')
    
    # Intentamos buscar el archivo de WOO (puedes cambiar este nombre)
    woo_file = None
    for f in os.listdir(data_dir):
        if 'woo' in f.lower() and f.endswith('.xlsx'):
            woo_file = os.path.join(data_dir, f)
            break
    
    all_data = []

    # 1. Procesar BD_POS
    if os.path.exists(pos_file):
        print(f"Procesando POS: {pos_file}")
        df_pos = pd.read_excel(pos_file)
        # Mapeo POS
        pos_unified = pd.DataFrame()
        pos_unified['fecha'] = pd.to_datetime(df_pos['FechaVenta'], errors='coerce')
        pos_unified['producto'] = df_pos['DescripcionProducto']
        pos_unified['venta'] = pd.to_numeric(df_pos['PrecioVentaConIGV_PEN'], errors='coerce').fillna(0)
        pos_unified['costo'] = pd.to_numeric(df_pos['CostoProductoPEN'], errors='coerce').fillna(0)
        pos_unified['plataforma'] = df_pos['Plataforma']
        pos_unified['fuente'] = 'POS'
        pos_unified['comision'] = 0
        pos_unified['envio'] = 0
        all_data.append(pos_unified)
    
    # 2. Procesar BD_WOO
    if woo_file and os.path.exists(woo_file):
        print(f"Procesando WOO: {woo_file}")
        # WooCommerce exports usually have title rows. We skip them to find the real header.
        df_woo = pd.read_excel(woo_file, header=3)
        
        # Mapping WOO (Basado en la captura)
        woo_unified = pd.DataFrame()
        
        # Clean current headers (sometimes they have spaces)
        df_woo.columns = [str(c).strip() for c in df_woo.columns]
        
        if 'Fecha' in df_woo.columns:
            woo_unified['fecha'] = pd.to_datetime(df_woo['Fecha'], errors='coerce')
        else:
            print(f"Error: No se encontró columna 'Fecha' en WOO. Columnas: {df_woo.columns.tolist()}")
            return

        woo_unified['producto'] = df_woo.get('Producto', df_woo.get('Artículos', 'Pedido Web'))
        woo_unified['venta'] = pd.to_numeric(df_woo['Venta (PEN)'], errors='coerce').fillna(0)
        woo_unified['costo'] = pd.to_numeric(df_woo['Costo (PEN)'], errors='coerce').fillna(0)
        
        # Fix for Plataforma if not present (WooCommerce is usually Web)
        woo_unified['plataforma'] = df_woo.get('Plataforma', 'Tienda Online')
        
        woo_unified['fuente'] = 'WOO'
        woo_unified['comision'] = pd.to_numeric(df_woo['Comisión (PEN)'], errors='coerce').fillna(0)
        woo_unified['envio'] = pd.to_numeric(df_woo['Envío (PEN)'], errors='coerce').fillna(0)
        all_data.append(woo_unified)
    else:
        print("Aviso: No se encontró archivo de BD_WOO en la carpeta data/")

    if not all_data:
        print("Error: No hay datos para migrar.")
        return

    # Consolidar
    df_master = pd.concat(all_data, ignore_index=True)
    df_master = df_master.dropna(subset=['fecha'])
    
    # --- ENRIQUECIMIENTO DE BI ---
    print("Enriqueciendo datos con dimensiones de tiempo y métricas...")
    
    # 1. Dimensiones de Tiempo
    df_master['anio'] = df_master['fecha'].dt.year
    df_master['mes'] = df_master['fecha'].dt.month
    df_master['dia'] = df_master['fecha'].dt.day
    df_master['dia_semana'] = df_master['fecha'].dt.day_name()
    df_master['semana_anio'] = df_master['fecha'].dt.isocalendar().week
    
    # 2. Categorización Automática de Productos
    def categorizar(nombre):
        nombre = str(nombre).upper()
        if 'CORREA' in nombre: return 'Correas'
        if 'CASE' in nombre or 'SILICONE' in nombre: return 'Cases'
        if 'MICA' in nombre or 'SCREEN' in nombre: return 'Micas/Protectores'
        if 'CARGADOR' in nombre or 'CABLE' in nombre: return 'Carga/Energía'
        if 'WATCH' in nombre or 'BAND' in nombre: return 'Smartwatches/Bands'
        return 'Otros'

    df_master['categoria'] = df_master['producto'].apply(categorizar)
    
    # 3. Métricas de Rentabilidad
    df_master['margen_unitario'] = df_master['venta'] - df_master['costo'] - df_master['comision']
    df_master['porcentaje_margen'] = (df_master['margen_unitario'] / df_master['venta'] * 100).replace([float('inf'), -float('inf')], 0).fillna(0)
    df_master['es_rentable'] = (df_master['margen_unitario'] > 0).astype(int)

    # Subir a Postgres
    print(f"Subiendo {len(df_master)} filas enriquecidas a la tabla 'ventas_master'...")
    try:
        df_master.to_sql('ventas_master', engine, if_exists='replace', index=False)
        print("¡Migración exitosa!")
    except Exception as e:
        print(f"Error al subir a Postgres: {e}")

if __name__ == "__main__":
    sync_data()
