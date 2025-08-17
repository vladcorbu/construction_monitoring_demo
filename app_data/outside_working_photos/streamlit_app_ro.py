import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Tablou de Bord pentru Observarea Șantierului",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stPlotlyChart {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 2px solid #cccccc;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏗️ Tablou de Bord pentru Observarea Șantierului")
st.markdown("Monitorizarea activității muncitorilor pe parcursul zilei cu vizualizări interactive")

# ---------------------------
# Data Loading & Preparation
# ---------------------------
@st.cache_data(show_spinner=False)
def discover_projects():
    """Descoperă toate proiectele disponibile în folderul app_data."""
    app_data_dir = 'app_data'
    projects = {}
    
    if not os.path.exists(app_data_dir):
        return projects
    
    # Caută fișiere JSON în app_data
    for filename in os.listdir(app_data_dir):
        if filename.endswith('_ro.json'):
            # Extrage numele proiectului (ex: inside_working_ro.json -> inside_working)
            project_name = filename.replace('_ro.json', '')
            json_path = os.path.join(app_data_dir, filename)
            photos_dir = os.path.join(app_data_dir, f"{project_name}_photos")
            
            # Verifică dacă folderul de fotografii există
            if os.path.exists(photos_dir):
                projects[project_name] = {
                    'json_file': json_path,
                    'photos_dir': photos_dir,
                    'display_name': project_name.replace('_', ' ').title()
                }
    
    return projects

@st.cache_data(show_spinner=False)
def load_project_data(json_path):
    """Încarcă datele pentru un proiect specific."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data, os.path.basename(json_path)
    except json.JSONDecodeError:
        st.error(f"❌ Nu s-a putut analiza {json_path} – JSON invalid.")
        return None, None
    except FileNotFoundError:
        st.error(f"❌ Fișierul {json_path} nu a fost găsit.")
        return None, None

@st.cache_data(show_spinner=False) 
def load_data():
    """Încarcă rezultatele JSON. Fișier principal: inside_construction_ro.json.
    Se revine la fișierele gemini de rezervă pentru compatibilitate retroactivă."""
    primary_files = [
        'inside_construction_ro.json',        # format nou în română
        'inside_construction.json',           # format nou în engleză
        'gemini_pro_results_second.json',    # format vechi
        'gemini_pro_results.json'            # format mai vechi
    ]
    for fp in primary_files:
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    return data, fp
                except json.JSONDecodeError:
                    st.error(f"❌ Nu s-a putut analiza {fp} – JSON invalid.")
                    continue
    st.error("❌ Nu s-a găsit niciun fișier de rezultate. Rulați scriptul de analiză mai întâi pentru a genera inside_construction_ro.json")
    return None, None

def dataframe_from_payload(payload: dict) -> pd.DataFrame:
    """Convertește payload-ul brut în DataFrame și derivă coloane auxiliare."""
    df = pd.DataFrame(payload['data'])
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])
    else:
        st.warning("'timestamp' lipsește din intrările de date – ordinea poate fi incorectă.")
        df['datetime'] = pd.to_datetime(df.index, unit='s')
    df['time'] = df['datetime'].dt.strftime('%H:%M')
    # Siguranță: asigură coloane numerice
    for col in ['working', 'idle', 'total']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    # Gestionează primary_task și task_category (format nou)
    if 'primary_task' in df.columns:
        # Convertește primary_task în lista de sarcini pentru compatibilitate retroactivă
        df['tasks'] = df['primary_task'].apply(lambda x: [x] if pd.notna(x) and x != '' else [])
        df['primary_task'] = df['primary_task'].fillna('necunoscut')
    else:
        # Format vechi: normalizează sarcinile în listă
        if 'tasks' in df.columns:
            df['tasks'] = df['tasks'].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))
        else:
            df['tasks'] = [[] for _ in range(len(df))]
        df['primary_task'] = df['tasks'].apply(lambda x: x[0] if x else 'necunoscut')
    
    # Gestionează task_category
    if 'task_category' not in df.columns:
        df['task_category'] = 'necunoscut'
    else:
        df['task_category'] = df['task_category'].fillna('necunoscut')
    
    # Procentul de productivitate
    df['productivity_pct'] = (df['working'] / df['total'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    return df

def available_tasks(df: pd.DataFrame) -> list:
    """Obține sarcinile disponibile din coloana primary_task sau tasks."""
    tasks = set()
    if 'primary_task' in df.columns:
        # Format nou: folosește primary_task
        tasks.update(df['primary_task'].dropna().unique())
    else:
        # Format vechi: folosește lista de sarcini
        for row in df['tasks']:
            tasks.update(row)
    # Elimină valorile goale/necunoscute
    tasks.discard('')
    tasks.discard('necunoscut')
    return sorted(tasks)

def available_task_categories(df: pd.DataFrame) -> list:
    """Obține categoriile de sarcini disponibile."""
    if 'task_category' in df.columns:
        categories = set(df['task_category'].dropna().unique())
        categories.discard('')
        categories.discard('necunoscut')
        return sorted(categories)
    return []

def filter_by_tasks(df: pd.DataFrame, selected_tasks: list, selected_categories: list = None) -> pd.DataFrame:
    """Filtrează dataframe-ul după sarcinile și/sau categoriile selectate."""
    if not selected_tasks and not selected_categories:
        return df
    
    mask = pd.Series([True] * len(df), index=df.index)
    
    if selected_tasks:
        if 'primary_task' in df.columns:
            # Format nou: filtrează după primary_task
            task_mask = df['primary_task'].isin(selected_tasks)
        else:
            # Format vechi: filtrează după lista de sarcini
            task_mask = df['tasks'].apply(lambda ts: any(t in ts for t in selected_tasks))
        mask = mask & task_mask
    
    if selected_categories and 'task_category' in df.columns:
        category_mask = df['task_category'].isin(selected_categories)
        mask = mask & category_mask
    
    return df[mask].copy()

def summarize_task_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """Returnează frecvența fiecărei sarcini și numărul ponderat de muncitori activi."""
    if 'primary_task' in df.columns:
        # Format nou: folosește primary_task
        task_df = df[df['primary_task'].notna() & (df['primary_task'] != 'necunoscut')].copy()
        if task_df.empty:
            return pd.DataFrame(columns=['task', 'frames', 'avg_working', 'avg_idle'])
        
        agg = task_df.groupby('primary_task').agg(
            frames=('filename', 'nunique'),
            avg_working=('working', 'mean'),
            avg_idle=('idle', 'mean')
        ).reset_index().rename(columns={'primary_task': 'task'}).sort_values('frames', ascending=False)
        return agg
    else:
        # Format vechi: folosește lista de sarcini
        rows = []
        for _, r in df.iterrows():
            for t in r['tasks']:
                rows.append({
                    'task': t,
                    'frame': r['filename'],
                    'working': r['working'],
                    'idle': r['idle']
                })
        if not rows:
            return pd.DataFrame(columns=['task', 'frames', 'avg_working', 'avg_idle'])
        task_df = pd.DataFrame(rows)
        agg = task_df.groupby('task').agg(
            frames=('frame', 'nunique'),
            avg_working=('working', 'mean'),
            avg_idle=('idle', 'mean')
        ).reset_index().sort_values('frames', ascending=False)
        return agg

def display_project_dashboard(data, data_file, photos_dir):
    """Afișează tabloul de bord pentru un proiect specific."""
    df = dataframe_from_payload(data)

    # Bara laterală: metadate și filtre pentru proiectul curent
    st.sidebar.header("⚙️ Controale")
    metadata = data.get('metadata', {})
    if metadata:
        with st.sidebar.expander("Metadate Analiză", expanded=True):
            st.write(f"**Fișier Sursă:** {data_file}")
            st.write(f"**Data:** {metadata.get('date', 'N/A')}")
            st.write(f"**Total Cadre:** {metadata.get('total_frames', len(df))}")
            st.write(f"**Descriere Activată:** {'✅' if metadata.get('description_enabled', False) else '❌'}")
            st.write(f"**Revizuire Activată:** {'✅' if metadata.get('review_enabled', False) else '❌'}")

    # Filtre pentru sarcini și categorii
    all_tasks = available_tasks(df)
    all_categories = available_task_categories(df)
    
    selected_tasks = []
    selected_categories = []
    
    if all_tasks:
        selected_tasks = st.sidebar.multiselect(
            "Filtrează după Tipul Sarcinii", options=all_tasks, default=[], key=f"tasks_{data_file}"
        )
    
    if all_categories:
        selected_categories = st.sidebar.multiselect(
            "Filtrează după Categoria Sarcinii", options=all_categories, default=[], key=f"categories_{data_file}"
        )
    
    filtered_df = filter_by_tasks(df, selected_tasks, selected_categories)
    if filtered_df.empty:
        st.warning("Niciun cadru nu se potrivește cu filtrul(filtrele) selectat(e). Se afișează datele nefiltrate pentru context.")
        filtered_df = df
    
    # Afișează revizuirea zilnică dacă este disponibilă
    if 'daily_review' in data:
        st.header("📝 Revizuire Zilnică")
        st.info(data['daily_review'])
    
    # Rândul de metrici
    st.header("📈 Metrici Cheie")
    col1, col2, col3, col4 = st.columns(4)
    
    base = filtered_df
    with col1:
        st.metric("Muncitori Medii (filtrate)", f"{base['total'].mean():.1f}")
    with col2:
        st.metric("În Lucru Medii", f"{base['working'].mean():.1f}")
    with col3:
        st.metric("Inactivi Medii", f"{base['idle'].mean():.1f}")
    with col4:
        avg_productivity = (base['working'].sum() / base['total'].sum() * 100) if base['total'].sum() > 0 else 0
        st.metric("Productivitate Generală", f"{avg_productivity:.1f}%")

    # ---------------------------------
    # Instantaneu Înainte și După Zi
    # ---------------------------------
    st.header("🪟 Înainte și După (Instantaneu Zi)")
    # Folosește df complet pentru extremitățile cronologice (nu filtrate) pentru a reflecta schimbarea întregii zile
    day_sorted = df.sort_values('datetime')
    first_row = day_sorted.iloc[0]
    last_row = day_sorted.iloc[-1]
    c1, c2 = st.columns(2)
    def show_frame(col, row, label):
        with col:
            st.subheader(label)
            img_path = os.path.join(photos_dir, row['filename'])
            if os.path.exists(img_path):
                st.image(Image.open(img_path), caption=f"{row['time']} | L:{row['working']} I:{row['idle']} T:{row['total']}", use_container_width=True)
            else:
                st.error(f"Imagine lipsă: {img_path}")
            
            # Afișează informații despre sarcină
            if 'primary_task' in row.index and pd.notna(row['primary_task']) and row['primary_task'] != 'necunoscut':
                st.caption(f"Sarcină: {row['primary_task']}")
                if 'task_category' in row.index and pd.notna(row['task_category']) and row['task_category'] != 'necunoscut':
                    st.caption(f"Categorie: {row['task_category']}")
            elif row.get('tasks'):
                st.caption("Sarcini: " + ", ".join(row['tasks']))
            
            if row.get('description'):
                with st.expander("Descriere"):
                    st.write(row['description'])
    show_frame(c1, first_row, "Începutul Zilei")
    show_frame(c2, last_row, "Sfârșitul Zilei")

    # Sumar rapid al diferenței
    delta_workers = last_row['total'] - first_row['total']
    delta_working = last_row['working'] - first_row['working']
    st.info(f"Schimbare de la început la sfârșit – Total Muncitori: {delta_workers:+}, În Lucru: {delta_working:+}, Productivitate Δ: {(last_row['productivity_pct'] - first_row['productivity_pct']):+.1f} pp")
    
    # Creează graficul interactiv
    st.header("📊 Cronologia Activității Muncitorilor")
    
    # Creează figura cu axa y secundară
    fig = go.Figure()
    
    # Adaugă trasee
    fig.add_trace(go.Scatter(
        x=base['datetime'],
        y=base['working'],
        name='În Lucru',
        line=dict(color='green', width=3),
        mode='lines+markers',
        marker=dict(size=8),
        hovertemplate='<b>În Lucru:</b> %{y}<br><b>Timp:</b> %{x|%H:%M}<br><b>Imagine:</b> %{text}<extra></extra>',
        text=base['filename']
    ))
    
    fig.add_trace(go.Scatter(
        x=base['datetime'],
        y=base['idle'],
        name='Inactivi',
        line=dict(color='orange', width=3),
        mode='lines+markers',
        marker=dict(size=8),
        hovertemplate='<b>Inactivi:</b> %{y}<br><b>Timp:</b> %{x|%H:%M}<br><b>Imagine:</b> %{text}<extra></extra>',
        text=base['filename']
    ))
    
    fig.add_trace(go.Scatter(
        x=base['datetime'],
        y=base['total'],
        name='Total',
        line=dict(color='blue', width=3, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8),
        hovertemplate='<b>Total:</b> %{y}<br><b>Timp:</b> %{x|%H:%M}<br><b>Imagine:</b> %{text}<extra></extra>',
        text=base['filename']
    ))
    
    # Actualizează aspectul
    fig.update_layout(
        title='Activitatea Muncitorilor Pe Parcursul Zilei',
        xaxis_title='Timp',
        yaxis_title='Numărul de Muncitori',
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Afișează graficul
    selected_points = st.plotly_chart(fig, use_container_width=True, key=f"timeline_{data_file}")
    
    # Graficul procentului de productivitate
    st.header("📊 Procentul de Productivitate în Timp")
    
    # Deja calculat în dataframe_from_payload
    
    fig_productivity = go.Figure()
    fig_productivity.add_trace(go.Scatter(
        x=base['datetime'],
        y=base['productivity_pct'],
        mode='lines+markers',
        line=dict(color='purple', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(128, 0, 128, 0.2)',
        hovertemplate='<b>Productivitate:</b> %{y:.1f}%<br><b>Timp:</b> %{x|%H:%M}<extra></extra>'
    ))
    
    fig_productivity.update_layout(
        title='Procentul de Productivitate Pe Parcursul Zilei',
        xaxis_title='Timp',
        yaxis_title='Productivitate (%)',
        yaxis=dict(range=[0, 100]),
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_productivity, use_container_width=True, key=f"productivity_{data_file}")
    
    # Secțiunea de vizualizare a imaginilor
    st.header("🖼️ Vizualizator Cadre")
    
    # Creează o casetă de selecție pentru selecția imaginii
    selected_time = st.selectbox(
        "Selectați un timp pentru a vizualiza imaginea șantierului:",
        options=base.index,
        format_func=lambda x: f"{base.loc[x, 'time']} - L:{base.loc[x, 'working']} I:{base.loc[x, 'idle']} T:{base.loc[x, 'total']}",
        key=f"frame_selector_{data_file}"
    )
    
    if selected_time is not None:
        selected_row = base.loc[selected_time]

        col1, col2 = st.columns([2, 1])

        with col1:
            # Afișează imaginea
            image_path = os.path.join(photos_dir, selected_row['filename'])
            if os.path.exists(image_path):
                image = Image.open(image_path)
                st.image(image, caption=f"Cadru la {selected_row['time']}", use_container_width=True)
            else:
                st.error(f"Imaginea nu a fost găsită: {image_path}")

        with col2:
            st.subheader("Detalii Cadru")
            st.metric("Timp", selected_row['time'])
            st.metric("În Lucru", selected_row['working'])
            st.metric("Inactivi", selected_row['idle'])
            st.metric("Total", selected_row['total'])
            productivity = (selected_row['working'] / selected_row['total'] * 100) if selected_row['total'] > 0 else 0
            st.metric("Productivitate", f"{productivity:.1f}%")

            # Afișează descrierea dacă este disponibilă
            if 'description' in selected_row and selected_row['description']:
                st.subheader("Descriere")
                st.write(selected_row['description'])
            
            # Afișează informații despre sarcină
            if 'primary_task' in selected_row.index and pd.notna(selected_row['primary_task']) and selected_row['primary_task'] != 'necunoscut':
                st.subheader("Informații Sarcină")
                st.write(f"**Sarcină Principală:** {selected_row['primary_task']}")
                if 'task_category' in selected_row.index and pd.notna(selected_row['task_category']) and selected_row['task_category'] != 'necunoscut':
                    st.write(f"**Categorie:** {selected_row['task_category']}")
            elif selected_row.get('tasks'):
                st.subheader("Sarcini")
                st.write(", ".join(selected_row['tasks']))

    # ---------------------------------
    # Grafice Suplimentare Orientate pe Inginerie
    # ---------------------------------
    st.header("📊 Analiză Suplimentară")
    add_col1, add_col2 = st.columns(2)

    # 1. Zonă suprapusă: În lucru vs Inactivi
    with add_col1:
        area_fig = go.Figure()
        area_fig.add_trace(go.Scatter(
            x=base['datetime'], y=base['working'], name='În Lucru', mode='lines',
            stackgroup='one', line=dict(width=0.5, color='green'), groupnorm=''
        ))
        area_fig.add_trace(go.Scatter(
            x=base['datetime'], y=base['idle'], name='Inactivi', mode='lines',
            stackgroup='one', line=dict(width=0.5, color='orange')
        ))
        area_fig.update_layout(
            title='Starea Muncii Suprapusă în Timp',
            xaxis_title='Timp', yaxis_title='Număr Muncitori', template='plotly_white', showlegend=True,
        )
        st.plotly_chart(area_fig, use_container_width=True, key=f"area_{data_file}")

    # 2. Graficul de bare pentru frecvența sarcinilor
    with add_col2:
        task_stats = summarize_task_frequencies(base)
        if not task_stats.empty:
            task_bar = px.bar(
                task_stats,
                x='task', y='frames',
                hover_data={'avg_working': ':.1f', 'avg_idle': ':.1f'},
                title='Frecvența Apariției Sarcinilor (Cadre)',
                labels={'frames': 'Cadre cu Sarcina', 'task': 'Sarcină'}
            )
            task_bar.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(task_bar, use_container_width=True, key=f"task_bar_{data_file}")
        else:
            st.info("Nu sunt disponibile date de sarcini pentru graficul de frecvență.")

    # 3. Productivitate vs Total Scatter cu tendință
    scatter_df = base[base['total'] > 0].copy()
    if not scatter_df.empty:
        scatter_fig = px.scatter(
            scatter_df,
            x='total', y='productivity_pct', text='time',
            title='Productivitate vs Dimensiunea Echipei',
            labels={'total': 'Muncitori Totali', 'productivity_pct': 'Productivitate (%)'}
        )
        if len(scatter_df) >= 2:
            # Potrivire liniară simplă
            coef = np.polyfit(scatter_df['total'], scatter_df['productivity_pct'], 1)
            xs = np.linspace(scatter_df['total'].min(), scatter_df['total'].max(), 50)
            ys = np.polyval(coef, xs)
            scatter_fig.add_traces([go.Scatter(x=xs, y=ys, mode='lines', name='Tendință', line=dict(color='purple', dash='dash'))])
        scatter_fig.update_traces(textposition='top center')
        st.plotly_chart(scatter_fig, use_container_width=True, key=f"scatter_{data_file}")

    # 4. Muncă productivă cumulativă (cadre-muncitor)
    cum_df = base.sort_values('datetime').copy()
    cum_df['cumulative_working'] = cum_df['working'].cumsum()
    cum_fig = go.Figure()
    cum_fig.add_trace(go.Scatter(
        x=cum_df['datetime'], y=cum_df['cumulative_working'], mode='lines+markers',
        name='În Lucru Cumulativ', line=dict(color='teal', width=3)
    ))
    cum_fig.update_layout(
        title='Numărul Cumulativ În Lucru (Unități Muncitor-Cadre)',
        xaxis_title='Timp', yaxis_title='Unități Cumulative În Lucru', template='plotly_white', height=400
    )
    st.plotly_chart(cum_fig, use_container_width=True, key=f"cumulative_{data_file}")
    
    # Sumarul statisticilor
    st.header("📊 Statistici Sumar")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistici Număr Muncitori")
        stats_df = pd.DataFrame({
            'Metrică': ['Medie', 'Max', 'Min', 'Dev. Standard'],
            'În Lucru': [
                    f"{base['working'].mean():.1f}",
                    f"{base['working'].max()}",
                    f"{base['working'].min()}",
                    f"{base['working'].std():.1f}"
            ],
            'Inactivi': [
                    f"{base['idle'].mean():.1f}",
                    f"{base['idle'].max()}",
                    f"{base['idle'].min()}",
                    f"{base['idle'].std():.1f}"
            ],
            'Total': [
                    f"{base['total'].mean():.1f}",
                    f"{base['total'].max()}",
                    f"{base['total'].min()}",
                    f"{base['total'].std():.1f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True)
    
    with col2:
        st.subheader("Orele de Vârf")
        # Găsește orele de vârf în datele filtrate/baza
        max_workers_idx = base['total'].idxmax()
        max_productivity_idx = base['productivity_pct'].idxmax()
        min_productivity_idx = base['productivity_pct'].idxmin()
        st.info(f"**Cei Mai Mulți Muncitori:** {base.loc[max_workers_idx, 'time']} ({base.loc[max_workers_idx, 'total']} muncitori)")
        st.success(f"**Productivitate Maximă:** {base.loc[max_productivity_idx, 'time']} ({base.loc[max_productivity_idx, 'productivity_pct']:.1f}%)")
        st.warning(f"**Productivitate Minimă:** {base.loc[min_productivity_idx, 'time']} ({base.loc[min_productivity_idx, 'productivity_pct']:.1f}%)")
    
    # Tabelul de date
    with st.expander("📋 Vizualizare Date Brute"):
        # Pregătește coloanele de afișare bazate pe datele disponibile
        display_cols = ['time', 'working', 'idle', 'total', 'productivity_pct']
        
        if 'primary_task' in base.columns:
            display_cols.extend(['primary_task', 'task_category'])
        elif 'tasks' in base.columns:
            display_cols.append('tasks')
            
        display_df = base[display_cols].copy()
        display_df['productivity_pct'] = display_df['productivity_pct'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_df, use_container_width=True)

# Aplicația principală
def main():
    # Descoperă toate proiectele disponibile
    projects = discover_projects()
    
    if not projects:
        st.error("❌ Nu s-au găsit proiecte în folderul app_data. Asigurați-vă că aveți fișiere JSON și foldere de fotografii corespunzătoare.")
        
        # Fallback la metoda veche pentru compatibilitatea inversă
        st.header("🔄 Mod de Compatibilitate")
        st.info("Se încearcă încărcarea din fișierele din directorul rădăcină...")
        data, data_file = load_data()
        if data:
            display_project_dashboard(data, data_file, 'inside_working_photos')
        else:
            st.error("Nu se poate încărca niciun proiect.")
        return
    
    # Creează file pentru fiecare proiect
    if len(projects) == 1:
        # Dacă există doar un proiect, nu afișa file
        project_name = list(projects.keys())[0]
        project_info = projects[project_name]
        st.header(f"🏗️ {project_info['display_name']}")
        
        data, data_file = load_project_data(project_info['json_file'])
        if data:
            display_project_dashboard(data, data_file, project_info['photos_dir'])
        else:
            st.error(f"Nu se pot încărca datele pentru proiectul {project_info['display_name']}")
    else:
        # Afișează file pentru mai multe proiecte
        tab_names = [projects[proj]['display_name'] for proj in projects.keys()]
        tabs = st.tabs(tab_names)
        
        for i, (project_name, project_info) in enumerate(projects.items()):
            with tabs[i]:
                st.header(f"🏗️ {project_info['display_name']}")
                
                data, data_file = load_project_data(project_info['json_file'])
                if data:
                    display_project_dashboard(data, data_file, project_info['photos_dir'])
                else:
                    st.error(f"Nu se pot încărca datele pentru proiectul {project_info['display_name']}")

if __name__ == "__main__":
    main()
