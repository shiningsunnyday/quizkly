import React from 'react';
import { ScrollView, FlatList, Text, StyleSheet, Button, View, SectionList } from 'react-native';
import { Question } from '../components/Questions/Question';

export default class TestScreen extends React.Component {

  static navigationOptions = {
    title: 'Test',
  };

  state = {
    currentDocTitle: "",
    currentDocText: "",
    currentDocChoices: [],
  }

  changeQuestion() {

    for(var i = 0; i < params.list.length - 1; i++) {
      let newQuestion = params.list[i+1];
      console.log(newQuestion.question, "is the next question in line!");
      this.setState({
        currentDocTitle: newQuestion.question,
        currentDocText: newQuestion.question,
      })
    }
  }

  render() {

    console.log("What I Got!");
    const params = this.props.navigation.state.params.params;

    return (
        // <FlatList style={styles.flatList} data={params.list}
        //   renderItem={({item, index, section}) =>
        //     <View style={styles.questionView} key={index}>
        //       <Text style={styles.questionText}>{item.question}</Text>
        //     </View>}
        // />
        <View style={styles.container}>
          <Button title="Next Question!" onPress={this.changeQuestion} style={styles.questionButton}/>
        </View>
    );
  }
}

const styles = StyleSheet.create({
  questionButton: {
    flex: 1,
    backgroundColor: 'yellow',
  },
  container: {
    flex: 1,
    flexDirection: 'column',
    justifyContent: 'center',
    backgroundColor: 'red',
  },
  flatList: {
    flex: 1,
    flexDirection: 'column',
    backgroundColor: 'green',
  },
  flatListContent: {
    flexDirection: 'column',
    backgroundColor: 'yellow',
    justifyContent: 'center',
    alignItems: 'center',
  },
  questionText: {
    flex: 1,
    flexDirection: 'row',
  },
  questionView: {
    flex: 1,
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'orange',
  },
});
