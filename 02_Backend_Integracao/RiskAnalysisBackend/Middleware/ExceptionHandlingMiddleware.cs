using RiskAnalysisBackend.Models;

namespace RiskAnalysisBackend.Middleware;

public sealed class ExceptionHandlingMiddleware
{
    private readonly RequestDelegate _proximo;
    private readonly ILogger<ExceptionHandlingMiddleware> _logger;

    public ExceptionHandlingMiddleware(RequestDelegate proximo, ILogger<ExceptionHandlingMiddleware> logger)
    {
        _proximo = proximo;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext contexto)
    {
        try
        {
            await _proximo(contexto);
        }
        catch (ArgumentException ex)
        {
            _logger.LogWarning("Requisição inválida: {Mensagem}", ex.Message);
            await Responder(contexto, StatusCodes.Status400BadRequest, new ErroResponse(ex.Message));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Erro não tratado em {Metodo} {Caminho}.", contexto.Request.Method, contexto.Request.Path);
            await Responder(contexto, StatusCodes.Status500InternalServerError,
                new ErroResponse("Erro interno ao processar a requisição."));
        }
    }

    private static Task Responder(HttpContext contexto, int status, ErroResponse corpo)
    {
        if (contexto.Response.HasStarted) return Task.CompletedTask;

        contexto.Response.Clear();
        contexto.Response.StatusCode = status;
        return contexto.Response.WriteAsJsonAsync(corpo);
    }
}
