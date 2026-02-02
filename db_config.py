# Configuración de base de datos PostgreSQL
# Reemplaza los valores con tus credenciales de pgAdmin

DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'db_izi2026',
    'user': 'postgres',
    'password': 'Bp2pvtMX%21%21%40%60ffP' # <--- CAMBIA ESTO
}

def get_db_url():
    return f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
 