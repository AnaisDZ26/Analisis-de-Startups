"""Nombres en español de los rubros de Crunchbase (category_code) y formateo de montos."""

ETIQUETAS = {
    "advertising": "Publicidad", "analytics": "Analítica de datos", "automotive": "Automotriz",
    "biotech": "Biotecnología", "cleantech": "Energías limpias (cleantech)", "consulting": "Consultoría",
    "design": "Diseño", "ecommerce": "E-commerce", "education": "Educación",
    "enterprise": "Software empresarial", "fashion": "Moda", "finance": "Finanzas",
    "games_video": "Videojuegos", "government": "Gobierno", "hardware": "Hardware",
    "health": "Salud y bienestar", "hospitality": "Hotelería", "legal": "Legal",
    "local": "Servicios locales", "manufacturing": "Manufactura", "medical": "Dispositivos médicos",
    "messaging": "Mensajería", "mobile": "Aplicaciones móviles", "music": "Música",
    "nanotech": "Nanotecnología", "network_hosting": "Redes y hosting", "news": "Noticias y medios",
    "nonprofit": "Sin fines de lucro", "other": "Otros", "pets": "Mascotas",
    "photo_video": "Foto y video", "public_relations": "Relaciones públicas",
    "real_estate": "Inmobiliario", "search": "Buscadores", "security": "Ciberseguridad",
    "semiconductor": "Semiconductores", "social": "Redes sociales", "software": "Software",
    "sports": "Deportes", "transportation": "Transporte", "travel": "Viajes", "web": "Web",
}


def etiqueta(codigo: str) -> str:
    return ETIQUETAS.get(codigo, codigo.replace("_", " ").capitalize())


def _coma(valor: float, decimales: int) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def fmt_usd(monto: float) -> str:
    """US$ 250 mil · US$ 1,5 millones · US$ 1 millón."""
    if monto >= 1e6:
        v = monto / 1e6
        txt = _coma(v, 0) if abs(v - round(v)) < 1e-9 else _coma(v, 1)
        return f"US$ {txt} {'millón' if txt == '1' else 'millones'}"
    v = monto / 1e3
    txt = _coma(v, 0) if abs(v - round(v)) < 1e-9 else _coma(v, 1)
    return f"US$ {txt} mil"


def fmt_pct(p: float, decimales: int = 1) -> str:
    return f"{_coma(100 * p, decimales)} %"
