import discord
from discord.ext import commands
import asyncio
import threading
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import requests
from typing import Optional, Dict, List

# Carregar variáveis de ambiente
load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# ==================== CONFIGURAÇÃO DO BOT DISCORD ====================

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ==================== CONFIGURAÇÃO DO FASTAPI ====================

app = FastAPI(title="Sistema de Recomendação Musical")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CARREGAR DADOS ====================

csv_path = "/home/container/documents/top50MusicFrom2010-2019.csv"
df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# Mapeamento de colunas
column_mapping = {
    'title': 'title',
    'artist': 'artist',
    'the genre of the track': 'genre',
    'year': 'year',
    'Beats.Per.Minute -The tempo of the song': 'bpm',
    'Energy- The energy of a song - the higher the value, the more energtic': 'energy',
    'Danceability - The higher the value, the easier it is to dance to this song': 'danceability',
    'Loudness/dB - The higher the value, the louder the song': 'loudness',
    'Liveness - The higher the value, the more likely the song is a live recording': 'liveness',
    'Valence - The higher the value, the more positive mood for the song': 'valence',
    'Length - The duration of the song': 'length',
    'Acousticness - The higher the value the more acoustic the song is': 'acousticness',
    'Speechiness - The higher the value the more spoken word the song contains': 'speechiness',
    'Popularity- The higher the value the more popular the song is': 'popularity'
}

df = df.rename(columns=column_mapping)
df_clean = df.drop_duplicates(subset=['title']).reset_index(drop=True)

# Características para similaridade
features = ['bpm', 'energy', 'danceability', 'loudness', 'liveness', 'valence', 'length', 'acousticness', 'speechiness', 'popularity']

# Pré-processamento
scaler = MinMaxScaler()
df_scaled = df_clean.copy()
df_scaled[features] = scaler.fit_transform(df_clean[features])

# Matriz de similaridade
similarity_matrix = cosine_similarity(df_scaled[features])

# Simulação de dados para Filtro Colaborativo
mock_interactions = {
    "user1": ["Hey, Soul Sister", "Love The Way You Lie", "TiK ToK"],
    "user2": ["Hey, Soul Sister", "Just the Way You Are", "Baby"],
    "user3": ["Love The Way You Lie", "Bad Romance", "Dynamite"],
    "user4": ["TiK ToK", "Bad Romance", "Secrets"],
    "user5": ["Just the Way You Are", "Baby", "Dynamite"]
}

# ==================== SPOTIFY API ====================

def get_spotify_token():
    """Obtém token de acesso do Spotify"""
    auth_url = "https://accounts.spotify.com/api/token"
    auth_data = {
        "grant_type": "client_credentials",
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET,
    }
    response = requests.post(auth_url, data=auth_data)
    if response.status_code == 200:
        return response.json()["access_token"]
    return None

def search_spotify_track(query: str):
    """Busca uma música no Spotify"""
    token = get_spotify_token()
    if not token:
        return None
    
    search_url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": query,
        "type": "track",
        "limit": 1
    }
    
    response = requests.get(search_url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["tracks"]["items"]:
            track = data["tracks"]["items"][0]
            return {
                "name": track["name"],
                "artist": track["artists"][0]["name"],
                "image": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
                "spotify_url": track["external_urls"]["spotify"],
                "id": track["id"]
            }
    return None

# ==================== ENDPOINTS FASTAPI ====================

@app.get("/recommendations/content-based/{song_title}")
async def content_based_recommendations(song_title: str, limit: int = 5):
    if song_title not in df_clean['title'].values:
        return {"error": "Música não encontrada"}
    
    idx = df_clean[df_clean['title'] == song_title].index[0]
    sim_scores = similarity_matrix[idx]
    similar_indices = sim_scores.argsort()[-(limit+1):-1][::-1]
    
    results = df_clean.iloc[similar_indices][['title', 'artist', 'genre', 'year', 'popularity']].to_dict(orient='records')
    return {"song_title": song_title, "recommendations": results}

@app.get("/recommendations/popular")
async def popular_recommendations(year: Optional[int] = None, genre: Optional[str] = None, limit: int = 5):
    filtered_df = df_clean.copy()
    
    if year:
        filtered_df = filtered_df[filtered_df['year'] == year]
    
    if genre:
        filtered_df = filtered_df[filtered_df['genre'].str.contains(genre, case=False, na=False)]
    
    results = filtered_df.sort_values(by='popularity', ascending=False).head(limit)
    return {"recommendations": results[['title', 'artist', 'year', 'genre', 'popularity']].to_dict(orient='records')}

@app.get("/search/spotify")
async def search_spotify(query: str):
    """Busca uma música no Spotify"""
    result = search_spotify_track(query)
    if result:
        return result
    return {"error": "Música não encontrada no Spotify"}

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "service": "music-recommendation-api"}

# ==================== COMANDOS DO BOT DISCORD ====================

@bot.event
async def on_ready():
    print(f"{bot.user} conectado ao Discord!")
    print(f"FastAPI rodando em http://localhost:8000")

@bot.command(name="recomendacao")
async def recomendacao(ctx, *, musica: str):
    """Busca recomendações para uma música"""
    async with ctx.typing():
        try:
            # Buscar no Spotify
            spotify_data = search_spotify_track(musica)
            
            if not spotify_data:
                await ctx.send(f"❌ Música '{musica}' não encontrada no Spotify")
                return
            
            # Buscar recomendações na API
            response = requests.get(
                f"http://localhost:8000/recommendations/content-based/{spotify_data['name']}",
                params={"limit": 5}
            )
            
            if response.status_code != 200:
                await ctx.send(f"❌ Erro ao buscar recomendações")
                return
            
            data = response.json()
            recommendations = data.get("recommendations", [])
            
            # Criar embed
            embed = discord.Embed(
                title=f"🎵 Recomendações para: {spotify_data['name']}",
                description=f"Artista: {spotify_data['artist']}",
                color=discord.Color.green()
            )
            
            if spotify_data.get("image"):
                embed.set_thumbnail(url=spotify_data["image"])
            
            for i, rec in enumerate(recommendations[:5], 1):
                embed.add_field(
                    name=f"{i}. {rec['title']}",
                    value=f"Artista: {rec['artist']}\nGênero: {rec['genre']}\nPopularidade: {rec['popularity']}",
                    inline=False
                )
            
            await ctx.send(embed=embed)
        
        except Exception as e:
            await ctx.send(f"❌ Erro: {str(e)}")

@bot.command(name="popular")
async def popular(ctx, ano: int = 2010):
    """Mostra as músicas mais populares de um ano"""
    async with ctx.typing():
        try:
            response = requests.get(
                "http://localhost:8000/recommendations/popular",
                params={"year": ano, "limit": 5}
            )
            
            if response.status_code != 200:
                await ctx.send(f"❌ Erro ao buscar músicas populares")
                return
            
            data = response.json()
            recommendations = data.get("recommendations", [])
            
            embed = discord.Embed(
                title=f"🔥 Músicas Populares de {ano}",
                color=discord.Color.orange()
            )
            
            for i, rec in enumerate(recommendations[:5], 1):
                embed.add_field(
                    name=f"{i}. {rec['title']}",
                    value=f"Artista: {rec['artist']}\nGênero: {rec['genre']}\nPopularidade: {rec['popularity']}",
                    inline=False
                )
            
            await ctx.send(embed=embed)
        
        except Exception as e:
            await ctx.send(f"❌ Erro: {str(e)}")

@bot.command(name="status")
async def status(ctx):
    """Verifica o status da API"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            await ctx.send("✅ API está online e funcionando!")
        else:
            await ctx.send("❌ API retornou erro")
    except:
        await ctx.send("❌ API está offline")

# ==================== RODAR BOT E API ====================

def run_fastapi():
    """Roda o FastAPI em uma thread separada"""
    uvicorn.run(app, host="0.0.0.0", port=22160, log_level="info")

def run_bot():
    """Roda o bot Discord"""
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    # Iniciar FastAPI em uma thread separada
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    # Aguardar um pouco para a API iniciar
    import time
    time.sleep(2)
    
    # Rodar o bot Discord
    run_bot()
