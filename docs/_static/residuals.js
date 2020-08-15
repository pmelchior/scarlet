function initResiduals(){
    var $div = $("#div-blends");

    for(var i=0; i<blendIds.length; i++){
        var bid = blendIds[i];
        var $div_header = $("<div></div>");
        var $div1 = $("<div id='div-blend-1-"+bid+"' class='div-residual'></div>");
        var $div2 = $("<div id='div-blend-2-"+bid+"' class='div-residual'></div>");

        // Add The blend header
        $div_header.append("<h2 id='header-"+bid+"' class='blend-id'>Blend "+bid+"</h2>");

        // Add the images
        $div1.append("<img id='img-1-"+bid+"' src='' alt='' class='residual-image'>");
        $div2.append("<img id='img-2-"+bid+"' src='' alt='' class='residual-image'>");

        $div.append($div_header);
        $div.append($div1);
        $div.append($div2);
    }
}

function initBranches(){
    var $select1 = $("#select-branch1");
    var $select2 = $("#select-branch2");

    // Load the residuals when the user changes branches
    $select1.change(function(){loadResiduals(1)});
    $select2.change(function(){loadResiduals(2)});

    for(var i=0; i<branches.length; i++){
        var branch = branches[i];
        $select1.append("<option value='"+branch+"'>"+branch+"</option>")
        $select2.append("<option value='"+branch+"'>"+branch+"</option>")
    }
}

function loadResiduals(index){
    const s3 = new AWS.S3();
    const branch = $("#select-branch"+index).val();
    console.log("branch", branch);

    for(i=0; i<blendIds.length; i++){
        var bid = blendIds[i];
        const objectKey = branch + "/"+bid+".png";
        console.log("object key", objectKey);
        const url = s3.getSignedUrl('getObject', {
            Bucket: "scarlet-residuals",
            Key: objectKey,
            Expires: 60,
        })
        console.log(url);
        $("#img-"+index+"-"+bid).attr("src", url);
    }
}
