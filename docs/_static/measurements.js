// Measurements for each branch
let measurements = [
    {}, // set 1
    {}, // set 2
    {}, // set3
];

// BoxChart for each measurement
let boxCharts = {};

// ScatterChart for each measurement
let scatterCharts = {};

let loadingBranches = [];

// Branches that have been added to the box plots
let addedBranches = [];

// Color of plots for merged branches
const mergedColor = "#2f7ed8";

// Color of plots for unmerged branches
const branchColor = "#910000";

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

// Default scaling for each plot
const logLookup = {
    "init time": "linear",
    "runtime": "linear",
    "iterations": "linear",
    "init logL": "logarithmic",
    "logL": "logarithmic",
    "g diff": "linear",
    "r diff": "linear",
    "i diff": "linear",
    "z diff": "linear",
    "y diff": "linear",
}

// Amount to multiply (Q3-Q1) by to get the whisker size
const whiskerFactor = 1.5;


// When the add button is clicked, add the currently selected branch to the plots
function onClickAdd(){
    let branch = $("#select-branch").val();
    // Only add the branch if it hasn't already been added
    if(!addedBranches.includes(branch)){
        addedBranches.push(branch);
        getBranchMeasurements(branch, function() {
            onLoadMeasurements(branch);
        });
    }
}


// Initialize the page by loading the list of all analyzed branches and
// merged branches from AWS
function initPage(){
    // Load the branches that have been processed by
    get_branches(function(){
        // Load the branches that have been merged
        get_merged_branches(function(){
            // Populate the drop downs
            initBranches(["select-branch"]);
            // Initialize the plots
            initPlots();
        });
    });
}


// Load measurements fom AWS for a given branch and the currently selected set
function getBranchMeasurements(branch, callback){
    let set_id = parseInt($("#select-dataset").val());
    if(!([1,2,3].includes(set_id))){
        alert("You must choose a valid dataset to plot");
    }

    // Only load measurements that haven't already been loaded
    if(!(branch in measurements[set_id-1])){
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
                measurements[set_id-1][branch] = data["Items"];
                callback(branch);
            }
        });
    } else {
        callback(branch);
    }
}


// Create the HTML elements that contain all of the measurement plots
// and initialize the box plots and scatter plots
function initPlots(){
    let $div = $("#div-plots");
    for(let i=0; i<plotMeasurements.length; i++){
        let measurement = plotMeasurements[i];
        let $measDiv = $("<div id='div-"+measurement+"' class=meas-div></div>");
        let $boxFig = $("<figure class='highcharts-figure'></figure>");
        let $boxPlot = $("<div id='div-"+measurement+"-box'></div>");
        let $scatterFig = $("<figure class='highcharts-figure'></figure>");
        let $scatterPlot = $("<div id='div-"+measurement+"-scatter'></div>");
        let $imgDiv = $("<div id='#'></div>");

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


// Create the box plot for a given measurement
function initBoxChart(measurement){
    boxCharts[measurement] = new Highcharts.chart("div-"+measurement+"-box", {
        chart: {
            type: 'boxplot',
            plotBorderColor: "#888888",
            plotBorderWidth: 2,
            zoomType: "y",
        },
        title: {
            text: measurement,
        },
        legend: {
            enabled: false
        },
        xAxis: {
            title: {
                text: 'Branch'
            },
            gridLineWidth: 0,
            labels: {
                rotation: -45,
            },
        },
        yAxis: {
            title: {
                text: yUnitLookup[measurement],
            },
            gridLineWidth: 0,
            type: logLookup[measurement],
        },
        series: [],
    });
}


// Create the scatter plot for a given measurement
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
            categories: [],
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
            type: logLookup[measurement],
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


// Combine all of the boxplot data into a series
function getBoxSeries(data){
    return {
        name: "Measurements",
        data: data,
        events: {
            click: function(event){
                populateScatterPlot(event.point.branch);
            }
        },
    }
}


// Combine all of the outlier data into a series
function getOutlierSeries(data){
    return {
        name: "Outliers",
        type: "scatter",
        data: data,
        events: {
            click: function(event){
                populateScatterPlot(event.point.branch);
            }
        },
        enableMouseTracking: false,
        tooltip: {
            enabled: false,
        },
    }
}


// When an unmerged measurement has been loaded from AWS update the charts
function onLoadMeasurements(branch){
    for(let i=0; i<plotMeasurements.length; i++){
        let measurement = plotMeasurements[i];
        let data = getBoxPlotData(branch, measurement, branchColor);
        let chart = boxCharts[measurement];

        // Add measurements to the plot
        let categories = chart.xAxis[0].categories.slice();
        categories.push(branch);
        chart.xAxis[0].setCategories(categories);
        chart.series[0].addPoint(data["measurements"], true);
        for(let j=0; j<data["outliers"].length; j++){
            chart.series[1].addPoint(data["outliers"][j], false);
        }
        chart.redraw();
    }


    // Update the scatter plots with the current branch
    populateScatterPlot(branch);
}


// When a merged measurement has been loaded from AWS update the charts
function onLoadMergedMeasurements(branch){
    let datapoints = Math.min(merged_branches.length, parseInt($("#input-history").val()));
    // If this is a branch that was just loaded, mark it as loaded
    let idx = loadingBranches.indexOf(branch);
    if(idx >= 0){
        loadingBranches.splice(idx, 1);
    }

    // Only update the plots once all of the measurements have been loaded
    if(loadingBranches.length === 0){
        for(let i=0; i<plotMeasurements.length; i++){
            let measurement = plotMeasurements[i];
            let chart = boxCharts[measurement];
            let categories = [];
            let series = [];
            let outliers = [];

            for(let j=datapoints-1; j>=0; j--){
                let _branch = merged_branches[j];
                let data = getBoxPlotData(_branch, measurement, mergedColor);
                categories.push(_branch);
                series.push(data["measurements"]);
                outliers.push(...data["outliers"]);
            }
            chart.addSeries(getBoxSeries(series));
            chart.addSeries(getOutlierSeries(outliers));

            // Set the x-axis for the chart
            chart.xAxis[0].setCategories(categories);
        }
    }
}


// Convert measurement data into data that can be inserted into a Highcharts series
function getBoxPlotData(branch, measurement, color){
    let meas = [];
    let set_id = $("#select-dataset").val();
    for(let i=0; i<measurements[set_id-1][branch].length; i++){
        meas.push(getYData(measurements[set_id-1][branch][i], measurement));
    }
    meas.sort(function(a, b){return a-b});

    let median = Math.round(meas.length/2);
    let idx = Math.floor(meas.length/4);

    let q1 = meas[idx];
    let q3 = meas[meas.length-idx];
    let whisker = whiskerFactor *  (q3-q1);
    let lowWhisker = Math.max(q1-whisker, meas[0]);
    let highWhisker = Math.min(q3+whisker, meas[meas.length-1]);

    let chart = boxCharts[measurement];
    let x = 0;
    if(chart.series.length>0){
        x = chart.series[0].data.length;
    }
    let datapoints = Math.min(merged_branches.length, parseInt($("#input-history").val()));
    if(merged_branches.includes(branch)){
        x = datapoints-1 - merged_branches.indexOf(branch);
    }

    let data = {
        low: lowWhisker,
        q1: meas[idx],
        median: meas[median],
        q3: meas[meas.length-idx],
        high: highWhisker,
        x: x,
        branch: branch,
        color: color,
    }


    let outliers = [];
    for(let i=0; i<meas.length; i++){
        let observation = meas[i];
        if(observation<lowWhisker || observation>highWhisker){
            outliers.push({
                x: x,
                y: observation,
                branch: branch,
                marker: {
                    fillColor: 'white',
                    lineWidth: 1,
                    lineColor: color,
                    symbol: "circle",
                },
            });
        }
    }

    return {
        measurements: data,
        outliers: outliers,
    };
}


// Populate the box plot with measurements from merged branches
function addMergedMeasurements(){
    console.log("calling addMergedMeasurements");
    // First clear all of the plot data
    for(let i=0; i<plotMeasurements.length; i++){
        let chart = boxCharts[plotMeasurements[i]];
        while(chart.series.length > 0)
            chart.series[0].remove(true);
        chart = scatterCharts[plotMeasurements[i]];
        chart.series[0].setData([]);
    }

    let datapoints = Math.min(merged_branches.length, parseInt($("#input-history").val()));
    loadingBranches = merged_branches.slice(0, datapoints);

    for(let i=datapoints-1; i>=0; i--){
        let branch = merged_branches[i];
        getBranchMeasurements(branch, function() {
            onLoadMergedMeasurements(branch);
        });
    }

    for(let i=0; i<addedBranches.length; i++){
        let branch = addedBranches[i];
        getBranchMeasurements(branch, function() {
            onLoadMeasurements(branch);
        });
    }
}


// Reset all of the scatter plots to use the measurements
// from a given branch.
function populateScatterPlot(branch){
    for(let j=0; j<plotMeasurements.length; j++){
        let measurement = plotMeasurements[j];
        let data = getScatterData(branch, measurement);
        let scatter = scatterCharts[measurement];
        scatter.series[0].setData(data);
        scatter.setTitle({text:measurement+": "+branch});
    }
}


// The the y-value for a given plot
function getYData(measData, measName){
    let y;
    if(measName.includes("diff")){
        let lhs = measData[measName[0]+" truth"];
        let rhs = measData[measName[0]+" mag"]
        y = lhs-rhs;
    } else {
        y = measData[measName];
    }
    return y;
}


// Convert measurement data into a format for a highcharts scatter plot
function getScatterData(branch, measurement){
    let set_id = $("#select-dataset").val();
    let meas = measurements[set_id-1][branch];
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


// A hack to copy the blend ID to the clipboard because
// javascript cannot do that natively
function textToClipboard(text) {
    let dummy = document.createElement("textarea");
    document.body.appendChild(dummy);
    dummy.value = text;
    dummy.select();
    document.execCommand("copy");
    document.body.removeChild(dummy);
}
