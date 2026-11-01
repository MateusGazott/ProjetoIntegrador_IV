using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using RiskAnalysisBackend.Configuration;
using RiskAnalysisBackend.Middleware;
using RiskAnalysisBackend.Models;
using RiskAnalysisBackend.Services;

const string PoliticaCors = "FrontendEChat";

var builder = WebApplication.CreateBuilder(args);

builder.Services.Configure<MLOptions>(builder.Configuration.GetSection(MLOptions.SectionName));

builder.Services
    .AddControllers()
    .ConfigureApiBehaviorOptions(opcoes =>
    {
        opcoes.InvalidModelStateResponseFactory = contexto =>
        {
            var detalhes = contexto.ModelState
                .Where(e => e.Value is { Errors.Count: > 0 })
                .SelectMany(e => e.Value!.Errors.Select(erro =>
                    string.IsNullOrWhiteSpace(erro.ErrorMessage) || erro.ErrorMessage.Contains("JSON", StringComparison.OrdinalIgnoreCase)
                        ? $"Valor inválido em '{e.Key}'."
                        : erro.ErrorMessage))
                .Distinct()
                .ToList();

            return new BadRequestObjectResult(new ErroResponse("Os dados enviados são inválidos.", detalhes));
        };
    });

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.AddSingleton(sp => ModelMetadataLoader.Carregar(sp.GetRequiredService<IOptions<MLOptions>>().Value));
builder.Services.AddSingleton<IRiskModel>(sp => RiskModelFactory.Criar(
    sp.GetRequiredService<IOptions<MLOptions>>().Value,
    sp.GetRequiredService<ModelMetadata>(),
    sp.GetRequiredService<ILoggerFactory>().CreateLogger("RiskModel")));
builder.Services.AddSingleton<IFeaturePreprocessor, FeaturePreprocessor>();
builder.Services.AddSingleton<IAttentionPointsAnalyzer, AttentionPointsAnalyzer>();
builder.Services.AddSingleton<IChatSummaryBuilder, ChatSummaryBuilder>();
builder.Services.AddSingleton<IDiagnosticHistoryStore, DiagnosticHistoryStore>();
builder.Services.AddSingleton<IRiskPredictionService, RiskPredictionService>();

var origens = builder.Configuration.GetSection($"{MLOptions.SectionName}:OrigensPermitidas").Get<string[]>() ?? Array.Empty<string>();
builder.Services.AddCors(opcoes => opcoes.AddPolicy(PoliticaCors, politica =>
{
    if (origens.Length > 0)
        politica.WithOrigins(origens).AllowAnyMethod().AllowAnyHeader();
    else
        politica.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader();
}));

var app = builder.Build();

app.Services.GetRequiredService<IRiskPredictionService>();

app.UseMiddleware<ExceptionHandlingMiddleware>();
app.UseSwagger();
app.UseSwaggerUI();
app.UseCors(PoliticaCors);
app.MapControllers();

app.Run();
