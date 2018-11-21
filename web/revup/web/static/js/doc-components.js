Vue.component('rev-ans', {
	props: ['ans','id', 'status'],
	template: '\
		<button class="ans-btn btn btn-block" :class="btnStatus" @click="click">\
			{{ans}}\
		</button>\
	',
	methods: {
		click: function(event) {
			this.$emit("click", this.id);
		}
	},
	computed: {
		btnStatus: function() {
			return {
				'btn-default': this.status == undefined || this.status == "default",
				'btn-success': this.status == "correct",
				'btn-danger':  this.status == "wrong"
			}
		}
	}
});

Vue.component('rev-qn', {
	props: ['qn'],
	template: '\
		<transition appear name="slide-fade">\
			<div class="panel panel-default">\
				<div class="panel-heading">\
					{{qn.text}}\
				</div>\
				<div class="panel-body">\
					<div class="col-md-6 col-sm-6" v-for="(ans, id) in qn.answers" :key="ans">\
						<rev-ans :ans="ans" :id="id" :status="getAnsStatus(id)" @click="ansClick">\
						</rev-ans>\
					</div>\
				</div>\
			</div>\
		</transition>\
	',
	data: function() {
		return {
			attempted: false,
			selected: null
		};
	},
	methods: {
		getAnsStatus: function(id) {
			if(this.selected!=null) {
				if(this.selected==id) {
					return id==this.qn.correctAns ? "correct" : "wrong";
				} else if(this.qn.correctAns==id) {
					return "correct";
				}
			}
			return "default";
		},
		ansClick: function(id) {
			if(this.selected!=null) {
				return;
			}
			this.selected = id;
			this.$emit("answer", {qn: this.qn.id, correct: this.qn.correctAns==id});
		}
	}
});

Vue.component('rev-qnstatus', {
	props: ['state'],
	template: '\
		<div class="alert" :class="alertType">{{ state.msg }}</div>\
	',
	computed: {
		alertType: function() {
			return {
				"alert-info": this.state.type=="info",
				"alert-danger": this.state.type=="error",
				"alert-success": this.state.type=="success",
			};
		}
	}
});

Vue.component('rev-doc', {
	props: ['doc'],
	template: '\
		<div>\
			<div class="row" v-for="(qn,ix) in doc.qns" :key="qn.id">\
				<rev-qn :qn="qn"></rev-qn>\
			</div>\
			<div class="row">\
				<rev-qnstatus :state="doc.state"></rev-qnstatus>\
			</div>\
		</div>\
	',
});
