namespace RiskAnalysisBackend.Configuration;

public class MLOptions
{
    public const string SectionName = "MLConfig";

    public string ModelPath { get; set; } = "ModelsML/modelo_risco.onnx";
    public string MetadataPath { get; set; } = "ModelsML/model_metadata.json";
    public float LimiteValorExtremo { get; set; } = 10000f;
    public int MaximoDiagnosticosEmMemoria { get; set; } = 500;
    public int ValidadeDiagnosticoMinutos { get; set; } = 120;
    public string[] OrigensPermitidas { get; set; } = Array.Empty<string>();

    public static string ResolverCaminho(string caminho) =>
        Path.IsPathRooted(caminho) ? caminho : Path.Combine(AppContext.BaseDirectory, caminho);
}
