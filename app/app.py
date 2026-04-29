from flask import Flask, render_template, request
import plotly.graph_objs as go
from plotly.offline import plot
from predict_lstm import get_lstm_predictions, SHORT_NAME_MAP, INDIAN_TICKERS
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

app = Flask(__name__)

def get_display_ticker(ticker):
    ticker = ticker.upper().strip()
    if ticker in SHORT_NAME_MAP:
        ticker = SHORT_NAME_MAP[ticker]
    return ticker  # return clean name without .NS for display

@app.route("/", methods=["GET", "POST"])
def index():
    ticker = None
    display_ticker = None

    if request.method == "POST":
        ticker = request.form.get("ticker").upper().strip()
        display_ticker = get_display_ticker(ticker)

    if ticker is None:
        return render_template(
            "index.html",
            ticker=None,
            graph="",
            metrics="",
            table=""
        )

    try:
        df_hist, df_pred = get_lstm_predictions(ticker=ticker, days_to_predict=30)
    except Exception as e:
        return render_template(
            "index.html",
            ticker=display_ticker,
            graph=f"<h3>Error: {str(e)}</h3>",
            metrics="",
            table=""
        )

    trace1 = go.Scatter(
        x=df_hist.index,
        y=df_hist.values.flatten(),
        mode="lines",
        name="Historical"
    )

    trace2 = go.Scatter(
        x=df_pred.index,
        y=df_pred["Predicted"],
        mode="lines+markers",
        name="Predicted",
        line=dict(dash="dash", color="red")
    )

    fig = go.Figure(
        data=[trace1, trace2],
        layout=go.Layout(
            title=f"{display_ticker} Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price"
        )
    )

    graph = plot(fig, output_type="div", config={
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawclosedpath", "drawcircle", "drawrect", "eraseshape"],
        "editable": True
    })

    actual = df_hist.values.flatten()[-len(df_pred):]
    predicted = df_pred["Predicted"].values

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    metrics = f"MAE: {mae:.2f} | RMSE: {rmse:.2f}"

    df_display = df_pred.copy()
    df_display.index = df_display.index.strftime("%Y-%m-%d")
    df_display.index.name = "Date"
    df_display.columns = ["Price"]
    df_display = df_display.reset_index()
    table_html = df_display.to_html(classes="pred-table", index=False)

    return render_template(
        "index.html",
        ticker=display_ticker,
        graph=graph,
        metrics=metrics,
        table=table_html
    )

if __name__ == "__main__":
    app.run(debug=True)