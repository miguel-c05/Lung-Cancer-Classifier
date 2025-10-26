import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

default_path = r"modelos\resultados_modelos.csv"
print(f"Default path: {default_path}")

def avaliar_resultados(csv_path: str = default_path):
    """
    Avalia as melhores combinações de datasets e verifica se há evolução
    em todas as métricas de teste e validação com o aumento da informação.
    Separa análise entre classificação binária e multi-classe.
    """
    # Verificar se o ficheiro existe
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ficheiro não encontrado: {csv_path}")
    
    # Ler ficheiro
    df = pd.read_csv(csv_path)
    
    # Identificar todas as métricas de teste e validação
    test_metrics = [col for col in df.columns if col.startswith('test_')]
    val_metrics = [col for col in df.columns if col.startswith('val_')]
    
    if not test_metrics and not val_metrics:
        raise ValueError("Nenhuma métrica de teste ou validação encontrada no dataset.")
    
    print(f"Métricas de teste encontradas: {test_metrics}")
    print(f"Métricas de validação encontradas: {val_metrics}")
      # Separar dados por tipo de classificação
    # Converter valores string para boolean se necessário
    if df['bin_malignancy'].dtype == 'object':
        df['bin_malignancy'] = df['bin_malignancy'].astype(str).map({'True': True, 'False': False})
        # Filtrar apenas registros com valores válidos (True/False)
        df = df[df['bin_malignancy'].notna()].copy()
    
    df_binary = df[df['bin_malignancy'] == True].copy()
    df_multiclass = df[df['bin_malignancy'] == False].copy()
    
    print(f"Registos classificação binária: {len(df_binary)}")
    print(f"Registos classificação multi-classe: {len(df_multiclass)}")
    
    # Adicionar coluna com o número de datasets combinados
    df_binary["n_datasets"] = df_binary["dataset"].apply(lambda x: len(str(x).split("||")))
    df_multiclass["n_datasets"] = df_multiclass["dataset"].apply(lambda x: len(str(x).split("||")))
      # Dicionários para armazenar resultados separados
    resultados = {
        "test": {"binary": {}, "multiclass": {}},
        "val": {"binary": {}, "multiclass": {}}
    }
    
    # Processar métricas de teste
    if test_metrics:
        if len(df_binary) > 0:
            print(f"\n{'='*80}")
            print("PROCESSANDO CLASSIFICAÇÃO BINÁRIA - MÉTRICAS DE TESTE")
            print(f"{'='*80}")
            resultados["test"]["binary"] = processar_metricas(df_binary, test_metrics, "BINÁRIA - TESTE")
        
        if len(df_multiclass) > 0:
            print(f"\n{'='*80}")
            print("PROCESSANDO CLASSIFICAÇÃO MULTI-CLASSE - MÉTRICAS DE TESTE")
            print(f"{'='*80}")
            resultados["test"]["multiclass"] = processar_metricas(df_multiclass, test_metrics, "MULTI-CLASSE - TESTE")
    
    # Processar métricas de validação
    if val_metrics:
        if len(df_binary) > 0:
            print(f"\n{'='*80}")
            print("PROCESSANDO CLASSIFICAÇÃO BINÁRIA - MÉTRICAS DE VALIDAÇÃO")
            print(f"{'='*80}")
            resultados["val"]["binary"] = processar_metricas(df_binary, val_metrics, "BINÁRIA - VALIDAÇÃO")
        
        if len(df_multiclass) > 0:
            print(f"\n{'='*80}")
            print("PROCESSANDO CLASSIFICAÇÃO MULTI-CLASSE - MÉTRICAS DE VALIDAÇÃO")
            print(f"{'='*80}")
            resultados["val"]["multiclass"] = processar_metricas(df_multiclass, val_metrics, "MULTI-CLASSE - VALIDAÇÃO")
    
    # Criar estrutura de pastas
    base_path = Path("metrics")
    base_path.mkdir(exist_ok=True)
    
    test_path = base_path / "test"
    val_path = base_path / "val"
    test_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)
    
    # Salvar evolução separada por tipo e métrica
    if resultados["test"]["binary"] and test_metrics:
        salvar_evolucao(resultados["test"]["binary"], test_metrics, test_path, "binary")
    
    if resultados["test"]["multiclass"] and test_metrics:
        salvar_evolucao(resultados["test"]["multiclass"], test_metrics, test_path, "multiclass")
        
    if resultados["val"]["binary"] and val_metrics:
        salvar_evolucao(resultados["val"]["binary"], val_metrics, val_path, "binary")
    
    if resultados["val"]["multiclass"] and val_metrics:
        salvar_evolucao(resultados["val"]["multiclass"], val_metrics, val_path, "multiclass")

    return resultados


def processar_metricas(df, test_metrics, tipo_classificacao):
    """
    Processa métricas para um tipo específico de classificação.
    """
    resultados_por_metrica = {}
    
    # Processar cada métrica de teste
    for metrica in test_metrics:
        print(f"\n{'='*80}")
        print(f"ANÁLISE DA MÉTRICA: {metrica.upper()} ({tipo_classificacao})")
        print(f"{'='*80}")
        
        # Obter os melhores resultados por combinação para esta métrica
        melhores = (
            df.loc[df.groupby("dataset")[metrica].idxmax(), ["dataset", metrica, "n_datasets"]]
            .sort_values(by=metrica, ascending=False)
            .reset_index(drop=True)
        )

        print(f"\n===== MELHORES RESULTADOS POR TAMANHO DE COMBINAÇÃO ({metrica}) =====")
        for n in sorted(melhores["n_datasets"].unique()):
            subset = melhores[melhores["n_datasets"] == n]
            top = subset.iloc[0]
            print(f"\nTop {n} dataset(s):")
            print(f"  Combinação: {top['dataset']}")
            print(f"  {metrica}: {top[metrica]:.4f}")

        # Verificar se há evolução com mais datasets
        medias = melhores.groupby("n_datasets")[metrica].mean().reset_index()
        medias["evolução"] = medias[metrica].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        print(f"\n===== EVOLUÇÃO MÉDIA POR TAMANHO DE COMBINAÇÃO ({metrica}) =====")
        print(medias)
        
        # Avaliação final: houve evolução global?
        evolucao_global = "Sim" if medias[metrica].iloc[-1] > medias[metrica].iloc[0] else "Não"
        print(f"\nEvolução global com mais informação: {evolucao_global}")
        
        # Armazenar resultados desta métrica
        resultados_por_metrica[metrica] = {
            'melhores': melhores,
            'medias': medias,
            'evolucao_global': evolucao_global
        }

    return resultados_por_metrica


def salvar_evolucao(resultados_por_metrica, test_metrics, base_path, suffix):
    """
    Salva a evolução das métricas em CSV.
    """
    # Salvar evolução de todas as métricas
    todas_evolucoes = None
    for metrica in test_metrics:
        if metrica in resultados_por_metrica:
            medias_metrica = resultados_por_metrica[metrica]['medias'].copy()
            medias_metrica.rename(columns={metrica: f'{metrica}_media', 'evolução': f'{metrica}_evolucao'}, inplace=True)
            
            if todas_evolucoes is None:
                todas_evolucoes = medias_metrica
            else:
                todas_evolucoes = todas_evolucoes.merge(medias_metrica, on='n_datasets', how='outer')
    
    if todas_evolucoes is not None:
        evolucao_completa_path = base_path / f"evolucao_metricas_{suffix}.csv"
        todas_evolucoes.to_csv(evolucao_completa_path, index=False)
        print(f"\nResultados {suffix} guardados em:")
        print(f"  - {evolucao_completa_path}")


def plot_metrics_evolution():
    """
    Cria gráficos mostrando a evolução das métricas ao longo do número de datasets.
    Mostra separadamente classificação binária e multi-classe para test e val.
    """
    base_path = Path("metrics")
    
    # Processar métricas de teste
    test_path = base_path / "test"
    if test_path.exists():
        print("\n" + "="*80)
        print("GERANDO GRÁFICOS PARA MÉTRICAS DE TESTE")
        print("="*80)
        plot_metrics_by_type(test_path, "TEST")
    
    # Processar métricas de validação
    val_path = base_path / "val"
    if val_path.exists():
        print("\n" + "="*80)
        print("GERANDO GRÁFICOS PARA MÉTRICAS DE VALIDAÇÃO")
        print("="*80)
        plot_metrics_by_type(val_path, "VALIDATION")


def plot_metrics_by_type(path, metric_type):
    """
    Plota gráficos para um tipo específico de métrica (TEST ou VALIDATION).
    """
    # Verificar quais ficheiros existem
    binary_path = path / "evolucao_metricas_binary.csv"
    multiclass_path = path / "evolucao_metricas_multiclass.csv"
    
    datasets_to_plot = []
    if binary_path.exists():
        datasets_to_plot.append(("binary", binary_path, f"Classificação Binária - {metric_type}"))
    if multiclass_path.exists():
        datasets_to_plot.append(("multiclass", multiclass_path, f"Classificação Multi-classe - {metric_type}"))
    
    if not datasets_to_plot:
        print(f"Aviso: Nenhum ficheiro de evolução {metric_type} encontrado.")
        return
    
    # Plotar cada tipo de dados separadamente
    for dataset_type, csv_path, title in datasets_to_plot:
        plot_single_classification_type(csv_path, title, dataset_type)
    
    # Se ambos existem, criar gráfico comparativo
    if len(datasets_to_plot) == 2:
        create_combined_comparison_plot(datasets_to_plot, metric_type)


def plot_single_classification_type(csv_path, title, dataset_type):
    """
    Plota gráficos para um único tipo de classificação.
    """
    # Ler dados
    df = pd.read_csv(csv_path)
      # Extrair nomes das métricas (colunas que terminam com '_media')
    metric_columns = [col for col in df.columns if col.endswith('_media')]
    metric_names = [col.replace('_media', '').replace('test_', '').replace('val_', '') for col in metric_columns]
    
    print(f"Gerando gráficos para {title}: {metric_names}")
    
    # Configurar cores
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_names)))
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Evolução das Métricas - {title}', fontsize=16, fontweight='bold')
    
    # Achatar array de axes para facilitar iteração
    axes_flat = axes.flatten()
    
    # Plotar cada métrica
    for i, (metric_col, metric_name, color) in enumerate(zip(metric_columns, metric_names, colors)):
        ax = axes_flat[i]
        
        # Plot da linha principal
        ax.plot(df['n_datasets'], df[metric_col], 
                marker='o', linewidth=2, markersize=8, 
                color=color, label=metric_name.title())
        
        # Destacar pontos de melhoria/piora com cores diferentes
        evolution_col = metric_col.replace('_media', '_evolucao')
        if evolution_col in df.columns:
            for j, (n_datasets, value, evolution) in enumerate(zip(df['n_datasets'], df[metric_col], df[evolution_col])):
                if evolution == 1:  # Melhoria
                    ax.scatter(n_datasets, value, color='green', s=100, alpha=0.7, marker='^')
                elif evolution == -1:  # Piora
                    ax.scatter(n_datasets, value, color='red', s=100, alpha=0.7, marker='v')
        
        # Configurações do gráfico
        ax.set_xlabel('Número de Datasets', fontweight='bold')
        ax.set_ylabel(f'{metric_name.title()}', fontweight='bold')
        ax.set_title(f'Evolução: {metric_name.replace("_", " ").title()}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(df['n_datasets'])
        
        # Configurar escala do eixo Y - 0 a 1 para todas as métricas exceto log_loss
        if 'log_loss' not in metric_name:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=max(0, df[metric_col].min() * 0.95))
        
        # Adicionar anotações dos valores
        for x, y in zip(df['n_datasets'], df[metric_col]):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
    
    # Remover subplot extra se necessário
    if len(metric_names) < len(axes_flat):
        fig.delaxes(axes_flat[-1])
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico
    output_path = Path(csv_path).parent / f"evolucao_metricas_{dataset_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico {title} salvo em: {output_path}")
    
    # Mostrar gráfico
    plt.show()


def create_combined_comparison_plot(datasets_to_plot, metric_type):
    """
    Cria gráficos comparativos entre classificação binária e multi-classe.
    """
    # Ler ambos os datasets
    dfs = {}
    for dataset_type, csv_path, title in datasets_to_plot:
        dfs[dataset_type] = pd.read_csv(csv_path)
    
    # Obter métricas comuns
    all_metrics = set()
    for df in dfs.values():
        metrics = [col.replace('_media', '').replace('test_', '').replace('val_', '') 
                  for col in df.columns if col.endswith('_media')]
        all_metrics.update(metrics)
    
    common_metrics = list(all_metrics)
    print(f"Criando gráfico comparativo para métricas {metric_type}: {common_metrics}")
    
    # Criar figura com subplots para cada métrica
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f'Comparação: Classificação Binária vs Multi-classe - {metric_type}', fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for i, metric_name in enumerate(common_metrics):
        if i >= len(axes_flat):
            break
            
        ax = axes_flat[i]
        
        # Plotar cada tipo de classificação
        for dataset_type, csv_path, title in datasets_to_plot:
            df = dfs[dataset_type]
            
            # Determinar o prefixo da métrica baseado no tipo
            prefix = 'test_' if metric_type == 'TEST' else 'val_'
            metric_col = f'{prefix}{metric_name}_media'
            
            if metric_col in df.columns:
                color = 'blue' if dataset_type == 'binary' else 'red'
                linestyle = '-' if dataset_type == 'binary' else '--'
                
                ax.plot(df['n_datasets'], df[metric_col], 
                       marker='o', linewidth=2, markersize=6,
                       color=color, linestyle=linestyle,
                       label=f'{metric_name.title()} - {title}')
        
        # Configurações do gráfico
        ax.set_xlabel('Número de Datasets', fontweight='bold')
        ax.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontweight='bold')
        ax.set_title(f'Comparação: {metric_name.replace("_", " ").title()}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Configurar escala do eixo Y - 0 a 1 para todas as métricas exceto log_loss
        if 'log_loss' not in metric_name:
            ax.set_ylim(0, 1)
    
    # Remover subplots extras
    for i in range(len(common_metrics), len(axes_flat)):
        fig.delaxes(axes_flat[i])
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico no diretório apropriado
    output_dir = Path(datasets_to_plot[0][1]).parent  # Diretório do primeiro arquivo CSV
    output_path = output_dir / f"comparacao_binary_vs_multiclass_{metric_type.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico comparativo {metric_type} binário vs multi-classe salvo em: {output_path}")
    plt.show()


def create_comparison_plot(df, metric_columns, metric_names, output_dir, dataset_type):
    """
    Cria um gráfico comparativo com todas as métricas para um tipo de classificação.
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_names)))
    
    for metric_col, metric_name, color in zip(metric_columns, metric_names, colors):
        plt.plot(df['n_datasets'], df[metric_col], 
                marker='o', linewidth=2, markersize=6, 
                label=metric_name.replace('_', ' ').title(), color=color)
    
    plt.xlabel('Número de Datasets', fontweight='bold', fontsize=12)
    plt.ylabel('Valor da Métrica', fontweight='bold', fontsize=12)
    plt.title(f'Comparação de Todas as Métricas - {dataset_type.title()}', fontweight='bold', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Configurar escala do eixo Y - 0 a 1 para todas as métricas exceto log_loss
    if 'log_loss' not in dataset_type:
        plt.ylim(0, 1)
    
    # Salvar gráfico comparativo
    comparison_path = output_dir / f"comparacao_metricas_{dataset_type}.png"
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico comparativo {dataset_type} salvo em: {comparison_path}")
    plt.show()


if __name__ == "__main__":
    try:
        print("Analisando resultados...")
        resultados = avaliar_resultados()
        
        print("\nGerando gráficos...")
        plot_metrics_evolution()
        
        print("\nAnálise completa!")
        
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print(f"Certifique-se de que o ficheiro existe no caminho especificado.")
        print(f"Caminho atual de trabalho: {os.getcwd()}")
    except Exception as e:
        print(f"Erro inesperado: {e}")