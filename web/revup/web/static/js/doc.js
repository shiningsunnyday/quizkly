(function() {
states = {
	LOADING: {
		type: "info",
		msg: "Loading..."
	},
	DONE: {
		type: "success",
		msg: "Done!"
	},
	ERROR: {
		type: "error",
		msg: "Error occurred!"
	}
};

//Look for /<id> at the end, with optional trailing /
var match = location.pathname.match(/\/(\d+)\/?$/);
var id = match == null ? null : match[1];

doc = { qns: [], state: states.LOADING, id: id };
app = new Vue({
	el: '#app',
	data: {
		doc: doc
	}
});

function transformQn(jsonQn) {
	var answers = _.shuffle(jsonQn.distractors);
	var ix = _.random(0,answers.length);
	answers.splice(ix,0,jsonQn.ans);
	return {
		id: jsonQn.id,
		text: jsonQn.qnText,
		answers: answers,
		//FIXME: index of correct answer may change on array modificiation. Bind id to answers, or fixup?
		//FIXME: or just store text.
		correctAns: ix
	};
}

function updateDoc() {
	if(doc.id==null) {
		doc.state = states.ERROR;
		stopUpdate();
		return;
	}
	var url = '/docjson/'+doc.id;
	if(doc.qns.length>0) {
		url += '?start=' + (doc.qns[doc.qns.length-1].id + 1);
	}
	$.get(url).then(function(data) {
		if(doc.qns.length>0) {
			var lastId = doc.qns[doc.qns.length-1].id;
			var ix = 0;
			for(;ix<data.qns.length;ix++) {
				if(data.qns[ix].id > lastId) {
					break;
				}
			}
			data.qns.splice(0,ix);
		}
		doc.qns = doc.qns.concat(data.qns.map(transformQn));
		
		switch(data.status) {
		case 1:
			doc.state = states.LOADING;
			break;
		case 2:
			doc.state = states.DONE;
			stopUpdate();
			break;
		case -1:
			doc.state = states.ERROR;
			stopUpdate();
			break;
		}
	});
}

updateTimer = setInterval(updateDoc,2000); //TODO Backoff

function stopUpdate() {
	clearInterval(updateTimer);
}

updateDoc();
})();
