// The client that connects to AWS DynamoDB
var docClient

// Blend ids for the current
var blendIds = [];

// Branches that have been analyzed
var branches = [];


function get_blends(set_id, callback){
    const params = {
        TableName: "scarlet_blend_ids",
        KeyConditionExpression: "#sid = :n",
        ExpressionAttributeNames: {
            "#sid": "set_id",
        },
        ExpressionAttributeValues: {
            ":n": set_id,
        }
    }

    docClient.query(params, function(err, data){
        var blend_data = data["Items"];
        if (err) {
            console.log(err);
        } else {
            // Store the blend id's
            blendIds = [];
            for(var i=0; i<blend_data.length; i++){
                blendIds.push(blend_data[i]["blend_id"]);
            }
            console.log("blend ids", blendIds);
            callback();
        }
    });
}


function get_branches(callback){
    var params = {
        TableName: "scarlet_branches",
    };

    docClient.scan(params, function(err, data){
        if (err) {
            console.log(err);
        } else {
            // Initialize the dropdown buttons
            var branch_data = data["Items"];
            for(var i=0; i<branch_data.length; i++){
                branches.push(branch_data[i]["branch"]);
            }
            console.log("branches", branches);
            callback();
        }
    });
}


function initBranches(options){
    for(var i=0; i<options.length; i++){
        var $select = $("#"+options[i]);
        for(var j=0; j<branches.length; j++){
            var branch = branches[j];
            $select.append("<option value='"+branch+"'>"+branch+"</option>")
        }
    }
}
