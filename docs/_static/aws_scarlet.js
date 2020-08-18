// The client that connects to AWS DynamoDB
let docClient

// Blend ids for the current
let blendIds = [];

// Branches that have been analyzed
let branches = [];

// Merged branches
let merged_branches = [];


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
        let blend_data = data["Items"];
        if (err) {
            console.log(err);
        } else {
            // Store the blend id's
            blendIds = [];
            for(let i=0; i<blend_data.length; i++){
                blendIds.push(blend_data[i]["blend_id"]);
            }
            console.log("blend ids", blendIds);
            callback();
        }
    });
}


function get_branches(callback){
    let params = {
        TableName: "scarlet_branches",
    };

    docClient.scan(params, function(err, data){
        if (err) {
            console.log(err);
        } else {
            // Initialize the dropdown buttons
            let branch_data = data["Items"];
            for(let i=0; i<branch_data.length; i++){
                branches.push(branch_data[i]["branch"]);
            }
            console.log("branches", branches);
            callback();
        }
    });
}


function get_merged_branches(callback){
    let params = {
        TableName: "scarlet_merged",
    };

    docClient.scan(params, function(err, data){
        if (err) {
            console.log(err);
        } else {
            // Initialize the dropdown buttons
            let branch_data = data["Items"];
            branch_data.sort(function(a,b){
                if(a["merge_order"] < b["merge_order"]){
                    return -1;
                } else if(a["merge_order"] > b["merge_order"]){
                    return 1;
                } else {
                    console.log("Invalid merge order, received duplicate values for", a, b);
                }
            })

            for(let i=0; i<branch_data.length; i++){
                let branch = branch_data[i]["branch"];
                // Only add branches that have been processed
                if(branches.includes(branch)){
                    merged_branches.push(branch);
                }
            }
            console.log("merged branches", merged_branches);
            callback();
        }
    });
}


function initBranches(options){
    for(let i=0; i<options.length; i++){
        let $select = $("#"+options[i]);
        for(let j=0; j<branches.length; j++){
            let branch = branches[j];
            // Show the unmerged branches first
            if(!merged_branches.includes(branch) && (branch !== "master")){
                $select.append("<option value='"+branch+"'>"+branch+"</option>")
            }
        }
        for(let j=0; j<merged_branches.length; j++){
            let branch = merged_branches[j];
            $select.append("<option value='"+branch+"'>"+branch+"</option>");
        }
    }
}
