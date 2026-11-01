using System.Globalization;
using System.Text;
using RiskAnalysisBackend.Models;

namespace RiskAnalysisBackend.Services;

public interface IChatSummaryBuilder
{
    string Construir(RiskDiagnosisResponse diagnostico, string? contextoUsuario);
}

public sealed class ChatSummaryBuilder : IChatSummaryBuilder
{
    private readonly ModelMetadata _metadata;

    public ChatSummaryBuilder(ModelMetadata metadata) => _metadata = metadata;

    public string Construir(RiskDiagnosisResponse diagnostico, string? contextoUsuario)
    {
        var cultura = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();

        sb.AppendLine("[DIAGNÓSTICO DO MODELO PREDITIVO]");
        sb.AppendLine($"- Empresa: {diagnostico.NomeEmpresa} (ID: {diagnostico.EmpresaId})");
        sb.AppendLine($"- Horizonte Avaliado: {diagnostico.HorizonteAnos} ano(s)");
        sb.AppendLine($"- Classificação de Risco: {diagnostico.ClassificacaoRisco}");
        sb.AppendLine($"- Probabilidade de Insolvência/Falência: {(diagnostico.ProbabilidadeInsolvencia * 100).ToString("0.##", cultura)}%");
        sb.AppendLine($"- Probabilidade de Saúde Financeira: {(diagnostico.ProbabilidadeSaudavel * 100).ToString("0.##", cultura)}%");
        sb.AppendLine($"- Indicadores efetivamente informados: {diagnostico.PercentualIndicadoresInformados.ToString("0.#", cultura)}%");
        sb.AppendLine($"- Modo de execução: {diagnostico.ModoExecucao}");

        if (!string.IsNullOrWhiteSpace(contextoUsuario))
            sb.AppendLine($"- Contexto Adicional Informado pelo Usuário: {contextoUsuario.Trim()}");

        if (diagnostico.AtributosImputados.Count > 0)
            sb.AppendLine($"- Observação de Pré-processamento: {diagnostico.AtributosImputados.Count} indicador(es) ausente(s) foram preenchidos com a mediana do treinamento ({string.Join(", ", diagnostico.AtributosImputados)}).");

        if (diagnostico.PontosDeAtencao.Count > 0)
        {
            sb.AppendLine("- Pontos de Atenção Identificados nos Indicadores:");
            foreach (string ponto in diagnostico.PontosDeAtencao)
                sb.AppendLine($"  * {ponto}");
        }

        if (diagnostico.AlertasValidacao.Count > 0)
            sb.AppendLine($"- Alertas de Validação: {string.Join(" | ", diagnostico.AlertasValidacao)}");

        sb.AppendLine("- Indicadores Financeiros Processados:");
        foreach (var (nome, valor) in diagnostico.IndicadoresProcessados)
        {
            string descricao = _metadata.FeatureDescriptions.TryGetValue(nome, out string? d) ? $" ({d})" : string.Empty;
            string origem = diagnostico.AtributosImputados.Contains(nome) ? " [mediana]" : string.Empty;
            sb.AppendLine($"  * {nome}{descricao}: {valor.ToString("0.0000", cultura)}{origem}");
        }

        return sb.ToString();
    }
}
