// Measurements for a branch
var measurements = {};

// BoxChart for each measurement
var boxCharts = {};

// ScatterChart for each measurement
var scatterCharts = {};

// Measurements that are plotted
const plotMeasurements = [
    "init time",
    "runtime",
    "iterations",
    "init logL",
    "logL",
    "g diff",
    "r diff",
    "i diff",
    "z diff",
    "y diff",
];

// Units for the y-axis
const yUnitLookup = {
    "init time": "time (ms)",
    "runtime": "time/source (ms)",
    "iterations": "iterations",
    "init logL": "logL",
    "logL": "logL",
    "g diff": "truth-model",
    "r diff": "truth-model",
    "i diff": "truth-model",
    "z diff": "truth-model",
    "y diff": "truth-model",
}

// Units for the scatter plot x-axis
const xUnitLookup = {
    "init time": "true r (mag)",
    "runtime": "true r (mag)",
    "iterations": "true r (mag)",
    "init-logL": "true r (mag)",
    "logL": "true r (mag)",
    "g diff": "true g (mag)",
    "r diff": "true r (mag)",
    "i diff": "true i (mag)",
    "z diff": "true z (mag)",
    "y diff": "true y (mag)",
}

// Column to use for x in the scatter plot
const xColLookup = {
    "init time": "r truth",
    "runtime": "r truth",
    "iterations": "r truth",
    "init logL": "r truth",
    "logL": "r truth",
    "g diff": "g truth",
    "r diff": "r truth",
    "i diff": "i truth",
    "z diff": "z truth",
    "y diff": "y truth",
}

const whiskerFactor = 1.5;


function get_branch_measurements(set_id, branch, callback){
    const params = {
        TableName: "scarlet_set"+set_id,
        KeyConditionExpression: "#br = :bname",
        ExpressionAttributeNames: {
            "#br": "branch",
        },
        ExpressionAttributeValues: {
            ":bname": branch,
        }
    }

    docClient.query(params, function(err, data){
        if (err) {
            console.log(err);
        } else {
            // Store the measurements
            measurements[branch] = data["Items"];
            console.log(branch+" measurements", measurements[branch]);
            callback();
        }
    });
}


function initPlots(){
    var $div = $("#div-plots");

    for(var i=0; i<plotMeasurements.length; i++){
        var measurement = plotMeasurements[i];
        var $measDiv = $("<div id='div-"+measurement+"' class=meas-div></div>");
        var $boxFig = $("<figure class='highcharts-figure'></figure>");
        var $boxPlot = $("<div id='div-"+measurement+"-box'></div>");
        var $scatterFig = $("<figure class='highcharts-figure'></figure>");
        var $scatterPlot = $("<div id='div-"+measurement+"-scatter'></div>");
        var $imgDiv = $("<div id='#'></div>");

        // Add the container for these measurements to the main div
        $div.append($measDiv);

        // Initialize the box plot
        $measDiv.append($boxFig);
        $boxFig.append($boxPlot);
        initBoxChart(measurement);

        // Initialize the scatter plot
        $measDiv.append($scatterFig);
        $scatterFig.append($scatterPlot);
        initScatterChart(measurement);

        // Initialize the image of the blend
        $measDiv.append($imgDiv);
    }
}


function getBoxPlotData(branch, measurement){
    var meas = [];
    for(var i=0; i<measurements[branch].length; i++){
        meas.push(getYData(measurements[branch][i], measurement));
    }
    meas.sort(function(a, b){return a-b});

    var median = Math.round(meas.length/2);
    var idx = Math.floor(meas.length/4);

    var q1 = meas[idx];
    var q3 = meas[meas.length-idx];
    var whisker = whiskerFactor *  (q3-q1);
    var lowWhisker = Math.max(q1-whisker, meas[0]);
    var highWhisker = Math.min(q3+whisker, meas[meas.length-1]);

    var data = {
        low: lowWhisker,
        q1: meas[idx],
        median: meas[median],
        q3: meas[meas.length-idx],
        high: highWhisker,
    }


    var outliers = [];
    for(var i=0; i<meas.length; i++){
        var observation = meas[i];
        if(observation<lowWhisker || observation>highWhisker){
            outliers.push(observation);
        }
    }
    return {
        measurements: data,
        outliers: outliers,
    };
}

function initBoxChart(measurement){
    boxCharts[measurement] = new Highcharts.chart("div-"+measurement+"-box", {
        chart: {
            type: 'boxplot',
            plotBorderColor: "#888888",
            plotBorderWidth: 2,
        },
        title: {
            text: measurement,
        },
        legend: {
            enabled: false
        },
        xAxis: {
            categories: [],
            title: {
                text: 'Branch'
            },
            gridLineWidth: 0,
        },
        yAxis: {
            title: {
                text: yUnitLookup[measurement],
            },
            gridLineWidth: 0,
        },
        series: [],
    });
}


function initScatterChart(measurement){
    let xUnit = xUnitLookup[measurement];
    let yUnit = yUnitLookup[measurement];

    scatterCharts[measurement] = new Highcharts.chart("div-"+measurement+"-scatter", {
        chart: {
            zoomType: "xy",
            plotBorderColor: "#888888",
            plotBorderWidth: 2,
        },
        title: {
            text: measurement,
        },
        legend: {
            enabled: false
        },
        xAxis: {
            title: {
                text: xUnit
            },
            gridLineWidth: 0,
        },
        yAxis: {
            title: {
                text: yUnit
            },
            gridLineWidth: 0,
        },
        tooltip: {
            useHTML: true,
            formatter: function(){
                return "Blend ID: "+this.point.blendId+"<br>"
                    + "Source index: "+this.point.sourceId+"<br>"
                    + "x: "+this.point.xPos+"<br>"
                    + "y: "+this.point.yPos+"<br>"
                    + xUnit+": "+this.x.toFixed(2)+"<br>"
                    + yUnit+": "+this.y.toFixed(2)+"<br>";
            }
        },
        series: [{
            type: "scatter",
            name: measurement,
            data: [],
            events: {
                click: function(event){
                    let blendId = event.point.blendId;
                    textToClipboard(blendId);
                    alert('saved "'+blendId+'" to the clipboard');
                }
            },
        }],
    });
}


function getYData(measData, measName){
    let y = 0;
    if(measName.includes("diff")){
        let lhs = measData[measName[0]+" truth"];
        let rhs = measData[measName[0]+" mag"]
        y = lhs-rhs;
    } else {
        y = measData[measName];
    }
    return y;
}


function getScatterData(branch, measurement){
    let meas = measurements[branch];
    let result = [];
    for(let i=0; i<meas.length; i++){
        let x = meas[i][xColLookup[measurement]];
        let y = getYData(meas[i], measurement);
        let blendId = meas[i]["meas_id"];
        result.push({
            x: x,
            y: y,
            blendId: blendId.slice(0, blendId.length-2),
            sourceId: blendId.slice(blendId.length-1),
            xPos: meas[i]["x"],
            yPos: meas[i]["y"],
            branch: branch,
        });
    }
    return result;
}


function textToClipboard(text) {
    var dummy = document.createElement("textarea");
    document.body.appendChild(dummy);
    dummy.value = text;
    dummy.select();
    document.execCommand("copy");
    document.body.removeChild(dummy);
}