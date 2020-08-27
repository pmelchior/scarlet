// Initialize all of the div's that show residual images
function initResiduals(){
    let $div = $("#div-blends");

    for(let i=0; i<blendIds.length; i++){
        let bid = blendIds[i];
        let $div_header = $("<div class='div-blend-header'></div>");
        let $div1 = $("<div id='div-blend-1-"+bid+"' class='div-residual'></div>");
        let $div2 = $("<div id='div-blend-2-"+bid+"' class='div-residual'></div>");

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

// Load and display the residual images
function loadResiduals(index){
    const s3 = new AWS.S3();
    const branch = $("#select-branch"+index).val();

    for(let i=0; i<blendIds.length; i++){
        let bid = blendIds[i];
        const objectKey = branch + "/"+bid+".png";
        const url = s3.getSignedUrl('getObject', {
            Bucket: "scarlet-residuals",
            Key: objectKey,
            Expires: 60,
        })
        $("#img-"+index+"-"+bid).attr("src", url);
    }
}
