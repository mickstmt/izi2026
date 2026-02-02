"""
Script de migración de datos de publicidad (META y TIKTOK) a PostgreSQL
Autor: Sistema IziStore 2026
Fecha: 2026-02-02
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import sys
import os

# Agregar el directorio raíz al path para importar db_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_config import DB_CONFIG

# Mapeo de meses en español a números
MESES_ESP = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
    'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
}

def create_ads_table(cursor):
    """Crea la tabla ads_costs en PostgreSQL si no existe"""

    drop_table_sql = """
    DROP TABLE IF EXISTS ads_costs CASCADE;
    """

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS ads_costs (
        id SERIAL PRIMARY KEY,
        plataforma VARCHAR(50) NOT NULL,
        anio INTEGER NOT NULL,
        mes INTEGER NOT NULL,
        mes_nombre VARCHAR(20),
        gasto_soles DECIMAL(10, 2) NOT NULL,
        venta_soles DECIMAL(10, 2) NOT NULL,
        roas DECIMAL(10, 4),
        fecha DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    # Índices para mejorar performance
    create_indexes_sql = """
    CREATE INDEX IF NOT EXISTS idx_ads_plataforma ON ads_costs(plataforma);
    CREATE INDEX IF NOT EXISTS idx_ads_fecha ON ads_costs(fecha);
    CREATE INDEX IF NOT EXISTS idx_ads_anio_mes ON ads_costs(anio, mes);
    """

    print("[*] Eliminando tabla ads_costs existente...")
    cursor.execute(drop_table_sql)

    print("[*] Creando tabla ads_costs...")
    cursor.execute(create_table_sql)

    print("[*] Creando indices...")
    cursor.execute(create_indexes_sql)

    print("[OK] Tabla ads_costs creada exitosamente\n")


def load_and_process_ads_data(file_path):
    """Carga y procesa los datos del archivo Excel de publicidad"""

    print(f"[*] Cargando datos desde: {file_path}")

    # Leer Excel
    df = pd.read_excel(file_path)

    print(f"   - Registros cargados: {len(df)}")
    print(f"   - Columnas: {df.columns.tolist()}\n")

    # Renombrar columnas para consistencia
    df.columns = ['plataforma', 'anio', 'mes_nombre', 'gasto_soles', 'venta_soles', 'roas']

    # Convertir mes en español a número
    df['mes'] = df['mes_nombre'].str.lower().map(MESES_ESP)

    # Validar que todos los meses se mapearon correctamente
    if df['mes'].isnull().any():
        print("[!] Advertencia: Algunos meses no se pudieron mapear:")
        print(df[df['mes'].isnull()][['mes_nombre']])

    # Crear columna de fecha (primer día del mes)
    df['fecha'] = pd.to_datetime(df[['anio', 'mes']].assign(dia=1).rename(
        columns={'anio': 'year', 'mes': 'month', 'dia': 'day'}
    ))

    # Limpiar datos
    df['gasto_soles'] = df['gasto_soles'].fillna(0).round(2)
    df['venta_soles'] = df['venta_soles'].fillna(0).round(2)
    df['roas'] = df['roas'].fillna(0).round(4)

    # Ordenar por fecha
    df = df.sort_values('fecha')

    print("[*] Procesamiento completado:")
    print(f"   - Plataformas: {df['plataforma'].unique()}")
    print(f"   - Anios: {sorted(df['anio'].unique())}")
    print(f"   - Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}\n")

    return df


def insert_ads_data(cursor, df):
    """Inserta los datos procesados en PostgreSQL"""

    print(f"[*] Insertando {len(df)} registros en ads_costs...")

    # Preparar datos para inserción
    columns = ['plataforma', 'anio', 'mes', 'mes_nombre', 'gasto_soles',
               'venta_soles', 'roas', 'fecha']

    data_tuples = [tuple(row) for row in df[columns].values]

    insert_sql = """
    INSERT INTO ads_costs (plataforma, anio, mes, mes_nombre, gasto_soles,
                          venta_soles, roas, fecha)
    VALUES %s
    """

    execute_values(cursor, insert_sql, data_tuples)

    print("[OK] Datos insertados exitosamente\n")


def print_summary_stats(cursor):
    """Muestra estadísticas resumidas de los datos migrados"""

    print("[*] RESUMEN DE DATOS MIGRADOS\n")
    print("=" * 60)

    # Total de registros
    cursor.execute("SELECT COUNT(*) FROM ads_costs")
    total_records = cursor.fetchone()[0]
    print(f"Total de registros: {total_records}")

    # Por plataforma
    cursor.execute("""
        SELECT plataforma,
               COUNT(*) as registros,
               ROUND(SUM(gasto_soles)::numeric, 2) as gasto_total,
               ROUND(SUM(venta_soles)::numeric, 2) as venta_total,
               ROUND(AVG(roas)::numeric, 2) as roas_promedio
        FROM ads_costs
        GROUP BY plataforma
        ORDER BY plataforma
    """)

    print("\n[*] Por Plataforma:")
    print("-" * 60)
    for row in cursor.fetchall():
        plat, regs, gasto, venta, roas = row
        print(f"{plat:10} | Registros: {regs:2} | Gasto: S/ {gasto:,.2f} | "
              f"Venta: S/ {venta:,.2f} | ROAS: {roas:.2f}x")

    # Por año
    cursor.execute("""
        SELECT anio,
               COUNT(*) as registros,
               ROUND(SUM(gasto_soles)::numeric, 2) as gasto_total,
               ROUND(SUM(venta_soles)::numeric, 2) as venta_total
        FROM ads_costs
        GROUP BY anio
        ORDER BY anio
    """)

    print("\n[*] Por Anio:")
    print("-" * 60)
    for row in cursor.fetchall():
        anio, regs, gasto, venta = row
        print(f"{anio} | Registros: {regs:2} | Gasto: S/ {gasto:,.2f} | Venta: S/ {venta:,.2f}")

    # Mejor mes por ROAS
    cursor.execute("""
        SELECT plataforma, anio, mes_nombre,
               ROUND(gasto_soles::numeric, 2) as gasto,
               ROUND(venta_soles::numeric, 2) as venta,
               ROUND(roas::numeric, 2) as roas
        FROM ads_costs
        WHERE roas > 0
        ORDER BY roas DESC
        LIMIT 5
    """)

    print("\n[*] Top 5 Mejores Meses por ROAS:")
    print("-" * 60)
    for row in cursor.fetchall():
        plat, anio, mes, gasto, venta, roas = row
        print(f"{plat} {mes} {anio} | ROAS: {roas:.2f}x | "
              f"Gasto: S/ {gasto:,.2f} | Venta: S/ {venta:,.2f}")

    print("\n" + "=" * 60)


def main():
    """Función principal de migración"""

    print("\n" + "=" * 60)
    print("[*] MIGRACION DE DATOS DE PUBLICIDAD - IZISTORE 2026")
    print("=" * 60 + "\n")

    # Ruta del archivo Excel
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, 'data', 'costos-ads.xlsx')

    if not os.path.exists(excel_path):
        print(f"[ERROR] No se encontro el archivo {excel_path}")
        sys.exit(1)

    conn = None
    cursor = None

    try:
        # Conectar a PostgreSQL
        print("[*] Conectando a PostgreSQL...")

        # Decodificar contraseña URL-encoded
        import urllib.parse
        password = urllib.parse.unquote(DB_CONFIG['password'])

        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=password
        )
        conn.autocommit = False
        cursor = conn.cursor()
        print("[OK] Conexion establecida\n")

        # Crear tabla
        create_ads_table(cursor)

        # Cargar y procesar datos
        df = load_and_process_ads_data(excel_path)

        # Insertar datos
        insert_ads_data(cursor, df)

        # Commit
        conn.commit()
        print("[OK] Transaccion confirmada (COMMIT)\n")

        # Mostrar resumen
        print_summary_stats(cursor)

        print("\n[OK] Migracion completada exitosamente\n")

    except Exception as e:
        print(f"\n[ERROR] Error durante la migracion: {str(e)}")
        if conn:
            conn.rollback()
            print("[*] Transaccion revertida (ROLLBACK)")
        sys.exit(1)

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        print("[*] Conexion cerrada\n")


if __name__ == "__main__":
    main()
