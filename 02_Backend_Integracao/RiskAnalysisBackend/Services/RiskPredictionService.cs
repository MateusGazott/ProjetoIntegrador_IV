using RiskAnalysisBackend.Models;

namespace RiskAnalysisBackend.Services;

public interface IRiskPredictionService
{
    RiskDiagnosisResponse AnalisarDadosFinanceiros(FinancialDataRequest request);
    ChatDiagnosticContextResponse? ConsultarDiagnosticoParaChat(string? empresaId = null);
    ModelMetadata ObterMetadadosModelo();
    ServiceStatusResponse ObterStatus();
}

public sealed class RiskPredictionService : IRiskPredictionService
{
    private readonly IRiskModel _modelo;
    private readonly ModelMetadata _metadata;
    private readonly IFeaturePreprocessor _preprocessador;
    private readonly IAttentionPointsAnalyzer _analisador;
    private readonly IChatSummaryBuilder _resumo;
    private readonly IDiagnosticHistoryStore _historico;
    private readonly ILogger<RiskPredictionService> _logger;

    public RiskPredictionService(
        IRiskModel modelo,
        ModelMetadata metadata,
        IFeaturePreprocessor preprocessador,
        IAttentionPointsAnalyzer analisador,
        IChatSummaryBuilder resumo,
        IDiagnosticHistoryStore historico,
        ILogger<RiskPredictionService> logger)
    {
        _modelo = modelo;
        _metadata = metadata;
        _preprocessador = preprocessador;
        _analisador = analisador;
        _resumo = resumo;
        _historico = historico;
        _logger = logger;
    }

    public ModelMetadata ObterMetadadosModelo() => _metadata;

    public ServiceStatusResponse ObterStatus() =>
        new("ok", _modelo.Modo, _modelo.EhModeloReal, _metadata.FeatureNames.Count, _historico.Quantidade);

    public RiskDiagnosisResponse AnalisarDadosFinanceiros(FinancialDataRequest request)
    {
        if (request == null)
            throw new ArgumentException("A requisição não pode ser nula.");

        if (request.Indicadores == null || request.Indicadores.Count == 0)
            throw new ArgumentException("Os dados financeiros (indicadores) da empresa não foram fornecidos.");

        if (request.HorizonteAnos < 1 || request.HorizonteAnos > 5)
            throw new ArgumentException("O horizonte de análise deve estar entre 1 e 5 anos.");

        var entrada = _preprocessador.Processar(request.Indicadores);
        var predicao = _modelo.Prever(entrada.Vetor, entrada.Processados);

        int total = _metadata.FeatureNames.Count;
        float percentualInformado = (total - entrada.Imputados.Count) * 100f / total;

        var diagnostico = new RiskDiagnosisResponse
        {
            EmpresaId = string.IsNullOrWhiteSpace(request.EmpresaId) ? Guid.NewGuid().ToString("N") : request.EmpresaId.Trim(),
            NomeEmpresa = string.IsNullOrWhiteSpace(request.NomeEmpresa) ? "Empresa Não Identificada" : request.NomeEmpresa.Trim(),
            HorizonteAnos = request.HorizonteAnos,
            DataAnalise = DateTime.UtcNow,
            ClassePrevista = predicao.Classe,
            ClassificacaoRisco = DeterminarFaixaDeRisco(predicao.ProbabilidadeInsolvencia),
            ProbabilidadeSaudavel = (float)Math.Round(predicao.ProbabilidadeSaudavel, 4),
            ProbabilidadeInsolvencia = (float)Math.Round(predicao.ProbabilidadeInsolvencia, 4),
            ModoExecucao = _modelo.Modo,
            PercentualIndicadoresInformados = (float)Math.Round(percentualInformado, 1),
            IndicadoresProcessados = entrada.Processados,
            AtributosImputados = entrada.Imputados,
            PontosDeAtencao = _analisador.Analisar(entrada.Processados, entrada.Imputados),
            AlertasValidacao = entrada.Alertas
        };

        var contexto = new ChatDiagnosticContextResponse
        {
            EmpresaId = diagnostico.EmpresaId,
            NomeEmpresa = diagnostico.NomeEmpresa,
            ClassificacaoRisco = diagnostico.ClassificacaoRisco,
            ProbabilidadeRiscoPercentual = (float)Math.Round(diagnostico.ProbabilidadeInsolvencia * 100, 2),
            ContextoInformadoPeloUsuario = string.IsNullOrWhiteSpace(request.ContextoUsuario) ? null : request.ContextoUsuario.Trim(),
            ResumoParaPromptLLM = _resumo.Construir(diagnostico, request.ContextoUsuario),
            DiagnosticoCompleto = diagnostico
        };

        _historico.Salvar(contexto);

        _logger.LogInformation(
            "Diagnóstico {EmpresaId}: {Classificacao} ({Prob:P1}) em modo {Modo}.",
            diagnostico.EmpresaId, diagnostico.ClassificacaoRisco, diagnostico.ProbabilidadeInsolvencia, diagnostico.ModoExecucao);

        return diagnostico;
    }

    public ChatDiagnosticContextResponse? ConsultarDiagnosticoParaChat(string? empresaId = null) =>
        _historico.Obter(empresaId);

    private string DeterminarFaixaDeRisco(float probabilidadeInsolvencia)
    {
        if (probabilidadeInsolvencia >= _metadata.ThresholdAltoRisco) return "Alto Risco";
        if (probabilidadeInsolvencia >= _metadata.ThresholdMedioRisco) return "Médio Risco";
        return "Baixo Risco";
    }
}
