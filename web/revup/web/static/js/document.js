var baseDelay = 1000;
var maxDelay = 10000;
var mult = 1.5;
var questionCount = 0;
var errorText = "<h2>Server error occurred.</h2>";
var emptyGenText = "<h2>No questions could be generated :(</h2>";
var doneText = "<h2>Thanks for taking the quiz!</h2>";
var loadText = "<h2>Loading...</h2>";
var qnTemplate = _.template(" \
                    <div class='qnContainer'> \
                        <div class='qn col-lg-6 col-lg-offset-3 text-center'> \
                            <br/> \
                            <h3><%- qnText %></h3> \
                            <br/> \
                        </div> \
                        <div class='ans col-lg-6 col-lg-offset-3 text-center' ></br> \
                            <% for(var i=0;i<options.length;i++) { %> \
                                <a class='purple ansOption' name='<%- \"ans\"+qnId %>' \
                                    data-correct='<%- options[i].correct %>' onClick='checkAnswer(this)'> \
                                    <span><%- options[i].text %></span> \
                                </a> \
                                </br> \
                            <% } %> \
                        </div> \
                    </div>");

function genQnHtml(qn) {
    var opts = [];
    opts.push({
        text: qn.ans,
        correct: true
    });
    for(var i=0;i<qn.distractors.length;i++) {
        opts.push({
            text: qn.distractors[i],
            correct: false
        });
    }
    return qnTemplate({
        options: _.shuffle(opts),
        qnText: qn.qnText,
        qnId: qn.id
    });
}

var delay = baseDelay;

function checkAnswer(btn){
    if($(btn).hasClass("disabled"))
        return;
    if(!$(btn).data("correct")) {
        $(btn).addClass("wrong");
        console.log("wrong!");
    } else {
        console.log("correct!");
    }
    $(btn).parent().find("[data-correct=true]").addClass("correct");
    $(btn).parent().find(".ansOption").addClass("disabled");
}

$(document).ready(function() {
    var docUrl = $("#docjsonUrl").val();

    $("#qncarousel").owlCarousel({
            singleItem:true,
            margin: 0,
            navigation: true,
            rewindNav: false
        });
    var carousel = $("#qncarousel").data('owlCarousel');


    function pullQuestions() {
        $.ajax({
            url: docUrl,
            data: {
                start: questionCount // TODO: Fix assumption that all qnNos are contiguous
            },
            type: 'GET',
            dataType: 'json',
            success: function(data) {
                var oldItem = carousel.owl.currentItem;
                var oldLen = carousel.owl.owlItems.length;
                carousel.removeItem();
                for(var i=0;i<data.qns.length;i++) {
                    var qn = data.qns[i];
                    var qnHtml = genQnHtml(qn);
                    carousel.addItem(qnHtml);
                }
                questionCount += data.qns.length;
                if(data.status==-1) {
                    carousel.addItem(errorText);
                } else if(data.status==2) {
                    if(questionCount==0)
                        carousel.addItem(emptyGenText);
                    else
                        carousel.addItem(doneText);
                } else {
                    if(data.qns.length==0) {
                        if(delay<maxDelay)
                            delay*=mult;
                    } else {
                        delay = baseDelay;
                    }
                    carousel.addItem(loadText);
                    setTimeout(pullQuestions,delay);
                }

                //Terrible hack to preserve current question when reloading
                if(oldItem==oldLen-1)
                    carousel.goTo(oldItem); //If at loading slide, slide in
                else
                    carousel.jumpTo(oldItem); //If in question, maintain
            },
            error: function() {
                if(delay<maxDelay)
                    delay*=mult;
                setTimeout(pullQuestions,delay);
            }
        })
    }
    carousel.addItem(loadText);
    pullQuestions();
    
});
